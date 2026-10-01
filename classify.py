"""Score each article of the CSV for relevance to a research question, as a percentage.

Backends:
  jev         OpenRouter Decisions API, one "noul" question per article; the
              probability is bucketed into 0-3 by quarters;
              % = score / 3 * 100.                   needs OPENROUTER_API_KEY
  nimble      same noul question via Ollama's /v1/systemone (OLLAMA_HOST, default :11434);
              % = noul probability * 100
  openrouter  chat completions with the 0-3 prompt;  % = score / 3 * 100

Usage:
  python classify.py --backend jev        --model typesafe/jev-1.13   --label jev
  python classify.py --backend openrouter --model qwen/qwen3.8-27b    --label qwen3.8_27b
  python classify.py --backend nimble     --model nimble              --label nimble
  python classify.py --merge   # build the comparison CSV from results/*.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, 'topic1_last7d.csv')
RESULTS_DIR = os.path.join(HERE, 'results')
MERGED_CSV = os.path.join(HERE, 'topic1_last7d_compare.csv')

QUESTION = ("Cet article traite-t-il du rôle de Taïwan dans les chaînes "
            "d'approvisionnement mondiales des technologies émergentes ?")

PROMPT = """Tu es analyste de veille documentaire. Tu dois évaluer la PERTINENCE de chacun des articles ci-dessous par rapport à une question de recherche précise.

QUESTION DE RECHERCHE :
{question}

BARÈME DE PERTINENCE (note de 0 à 3) :
- 3 = CENTRAL : la question de recherche est le sujet principal de l'article. Retirer ce thème de l'article le viderait de sa substance.
- 2 = SUBSTANTIEL : le thème est traité de façon développée (plusieurs paragraphes, une analyse, des faits précis) sans être le sujet principal de l'article.
- 1 = MENTION : le thème n'est évoqué qu'en passant — une phrase, une citation isolée, une allusion.
- 0 = HORS SUJET : l'article ne traite pas de ce thème, même indirectement.

RÈGLES IMPÉRATIVES :
1. Tu dois rendre un verdict pour CHACUN des {n_articles} articles fournis, sans aucune exception. Un article difficile à juger reçoit une note, jamais un silence.
2. Recopie l'identifiant de l'article EXACTEMENT tel qu'il est écrit dans l'en-tête (champ « id »). Ne le reformate pas, ne l'abrège pas, ne le traduis pas.
3. Juge uniquement le contenu fourni. N'utilise aucune connaissance extérieure sur le sujet ou sur la publication.
4. Un article peut parler du pays ou des entreprises concernés sans traiter la question posée : dans ce cas la note est basse. C'est le THÈME de la question qui est évalué, pas la présence de mots-clés.
5. La justification est en français, factuelle, et fait 20 mots au maximum.
6. N'ajoute aucun commentaire, aucun préambule et aucun texte hors du JSON.

=== ARTICLES À ÉVALUER ({n_articles}) ===
{corpus}

=== RÉPONSE ATTENDUE ===
Réponds UNIQUEMENT par un objet JSON valide conforme à ce schéma :
{schema}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 3},
                    "justification": {"type": "string"},
                },
                "required": ["id", "score", "justification"],
            },
        }
    },
    "required": ["verdicts"],
}


def load_articles():
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_prompt(articles):
    corpus = '\n\n'.join(
        f"--- ARTICLE ---\nid: {a['article_id']}\ntitre: {a['title']}\n"
        f"date: {a['date']}\n\n{a['body_text']}"
        for a in articles)
    return PROMPT.format(question=QUESTION, n_articles=len(articles), corpus=corpus,
                         schema=json.dumps(SCHEMA, ensure_ascii=False, indent=2))


def call_openrouter(prompt, model):
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        sys.exit('OPENROUTER_API_KEY is not set')
    resp = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'X-Title': 'jev-tests'},
        json={'model': model,
              'messages': [{'role': 'user', 'content': prompt}],
              'temperature': 0,
              'response_format': {'type': 'json_object'},
              'usage': {'include': True}},
        timeout=600)
    resp.raise_for_status()
    data = resp.json()
    if 'choices' not in data:  # upstream failures are reported in-body
        raise RuntimeError(f'OpenRouter error: {data}')
    print(f"  model used: {data.get('model')}  usage: {data.get('usage')}")
    return data['choices'][0]['message']['content']


def decision_endpoint(backend):
    """URL + headers for the Jev-style decisions API (state + typed questions)."""
    if backend == 'jev':
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            sys.exit('OPENROUTER_API_KEY is not set')
        return ('https://openrouter.ai/api/alpha/decisions',
                {'Authorization': f'Bearer {key}', 'X-Title': 'jev-tests'})
    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')
    if not host.startswith('http'):
        host = 'http://' + host
    return f'{host}/v1/systemone', {}


def ask_noul(url, headers, article, model):
    resp = requests.post(
        url, headers=headers,
        json={'model': model,
              'state': {'title': article['title'], 'body': article['body_text']},
              'questions': {'relevant': {'type': 'noul', 'instructions': QUESTION}}},
        timeout=300)
    if not resp.ok:  # the body says why (e.g. input too long), raise_for_status hides it
        raise requests.HTTPError(f'{resp.status_code}: {resp.text[:500]}', response=resp)
    data = resp.json()
    return data['answers']['relevant']['noul'], data


def noul_to_score(noul):
    """Bucket a noul probability into the 0-3 scale: [0, .25) -> 0, ..., [.75, 1] -> 3."""
    return min(int(noul * 4), 3)


def classify_noul(backend, model, articles):
    url, headers = decision_endpoint(backend)
    results, cost = {}, 0.0
    for i, a in enumerate(articles, 1):
        try:
            noul, data = ask_noul(url, headers, a, model)
        except (KeyError, requests.RequestException) as exc:
            print(f'  [{i}/{len(articles)}] {a["article_id"]}: failed: {exc}')
            continue
        cost += (data.get('usage') or {}).get('cost', 0) or 0
        method = f"{backend}/{data.get('model', model)}"
        if backend == 'jev':
            score = noul_to_score(noul)
            results[a['article_id']] = {'raw': score, 'pct': to_pct(score),
                                        'reason': f'noul={noul:.2f}', 'method': method}
            print(f'  [{i}/{len(articles)}] {score} ({noul:.2f})  {a["title"][:70]}')
        else:
            results[a['article_id']] = {'raw': noul, 'pct': round(noul * 100, 1), 'reason': '',
                                        'method': method}
            print(f'  [{i}/{len(articles)}] {noul:.2f}  {a["title"][:70]}')
    if cost:
        print(f'  total cost: ${cost:.6f}')
    return results


def parse_verdicts(text):
    # Reasoning models sometimes wrap the JSON in <think> blocks or code fences.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    start, end = text.find('{'), text.rfind('}')
    data = json.loads(text[start:end + 1])
    return {v['id'].strip(): v for v in data.get('verdicts', [])}


def classify(backend, model, articles, retries=2):
    verdicts, todo = {}, articles
    for attempt in range(retries + 1):
        print(f'  attempt {attempt + 1}: {len(todo)} articles')
        t0 = time.time()
        try:
            got = parse_verdicts(call_openrouter(build_prompt(todo), model))
        except (json.JSONDecodeError, ValueError, RuntimeError, requests.RequestException) as exc:
            print(f'  failed: {exc}')
            got = {}
        print(f'  {len(got)} verdicts in {time.time() - t0:.1f}s')
        wanted = {a['article_id'] for a in todo}
        verdicts.update({k: v for k, v in got.items() if k in wanted})
        todo = [a for a in todo if a['article_id'] not in verdicts]
        if not todo:
            break
    if todo:
        print(f'  WARNING: no verdict for {len(todo)} articles')
    return verdicts


def to_pct(score):
    return round(int(score) / 3 * 100, 1)


def run(args):
    articles = load_articles()
    print(f'{args.label}: {args.backend}/{args.model}')
    if args.backend in ('jev', 'nimble'):
        results = classify_noul(args.backend, args.model, articles)
    else:
        results = {k: {'raw': v['score'], 'pct': to_pct(v['score']),
                       'reason': v.get('justification', ''),
                       'method': f'{args.backend}/{args.model}'}
                   for k, v in classify(args.backend, args.model, articles).items()}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f'{args.label}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['article_id', 'score_raw', 'score_pct', 'reason', 'method'])
        for a in articles:
            r = results.get(a['article_id'], {})
            w.writerow([a['article_id'], r.get('raw', ''), r.get('pct', ''),
                        r.get('reason', ''), r.get('method', '')])
    print(f'  wrote {out}')


def merge():
    articles = load_articles()
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*.csv'))):
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline='', encoding='utf-8') as f:
            runs[label] = {r['article_id']: r for r in csv.DictReader(f)}
    cols = ['article_id', 'title', 'url', 'date', 'score_haiku_pct', 'reason_haiku']
    for label in runs:
        cols += [f'score_{label}_pct', f'reason_{label}']
    cols.append('body_text')
    with open(MERGED_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in articles:
            row = {k: a[k] for k in ('article_id', 'title', 'url', 'date', 'body_text')}
            row['score_haiku_pct'] = to_pct(a['score'])
            row['reason_haiku'] = a['reason']
            for label, res in runs.items():
                r = res.get(a['article_id'], {})
                row[f'score_{label}_pct'] = r.get('score_pct', '')
                row[f'reason_{label}'] = r.get('reason', '')
            w.writerow(row)
    print(f'wrote {MERGED_CSV} ({len(articles)} rows, runs: haiku, {", ".join(runs)})')
    # Quick agreement summary vs haiku.
    for label, res in runs.items():
        pairs = [(to_pct(a['score']), float(res[a['article_id']]['score_pct']))
                 for a in articles if res.get(a['article_id'], {}).get('score_pct')]
        if pairs:
            mae = sum(abs(h - m) for h, m in pairs) / len(pairs)
            print(f'  {label}: {len(pairs)} scored, mean abs diff vs haiku {mae:.1f} pts')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['jev', 'nimble', 'openrouter'])
    p.add_argument('--model')
    p.add_argument('--label')
    p.add_argument('--merge', action='store_true')
    a = p.parse_args()
    if a.merge:
        merge()
    elif a.backend and a.model and a.label:
        run(a)
    else:
        p.error('give --backend, --model and --label, or --merge')
