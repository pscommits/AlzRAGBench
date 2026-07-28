# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "networkx",
#     "huggingface_hub",
# ]
# ///
"""
Alzheimer's Domain Knowledge Graph — build once, query forever. (HF Job version)

Harvests Alzheimer's abstracts from PubMed, extracts ALL relations from each
abstract in a single LLM call, merges them into one dense graph, and measures
coverage on the Alzheimer's questions in MIRAGE.

Why a domain KG beats a per-question KG: one call per abstract yields many
edges (vs one call per entity PAIR yielding one edge), the graph is built once
instead of per question, and entities merge across papers so relations backed
by multiple studies get stronger.

The headline number it prints: BRIDGE-fact coverage on the AD questions —
directly comparable to the 0% Hetionet gave on PubMedQA*/BioASQ.

Run (cheapest sensible setup, ~1.1h, ~$0.45):
    hf jobs uv run --flavor t4-small --timeout 3h \
        --secrets HF_TOKEN \
        --env KG_REPO=NathanPereira/alzheimers-kg \
        --env NCBI_EMAIL=you@example.com \
        --env N_ABSTRACTS=300 \
        alzheimers_kg_job.py

Resume / densify: rerun with a larger N_ABSTRACTS. The graph is pulled from
KG_REPO, already-processed PMIDs are skipped, and new edges merge in.
"""
import os, re, json, time, pickle, urllib.parse, urllib.request
from collections import Counter
from difflib import get_close_matches
import torch
import networkx as nx

# ----------------------------- config ---------------------------------------
DISEASE      = os.environ.get("DISEASE", "Alzheimer's disease")
N_ABSTRACTS  = int(os.environ.get("N_ABSTRACTS", "300"))   # = number of LLM calls
KG_REPO      = os.environ.get("KG_REPO", "NathanPereira/alzheimers-kg")
GENERATOR    = os.environ.get("GENERATOR", "google/medgemma-4b-it")
CONF_MIN     = int(os.environ.get("CONF_MIN", "6"))
NCBI_EMAIL   = os.environ.get("NCBI_EMAIL", "example@example.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
CHECKPOINT_N = int(os.environ.get("CHECKPOINT_N", "50"))
MAX_NEW_TOK  = int(os.environ.get("MAX_NEW_TOKENS", "256"))  # was 420 — dominates runtime
PRINT_EVERY  = int(os.environ.get("PRINT_EVERY", "1"))       # progress line frequency
FRESH_START  = os.environ.get("FRESH_START", "0") == "1"     # ignore existing graph, rebuild
HF_TOKEN     = os.environ.get("HF_TOKEN")

# question filter — widen/narrow to change which MIRAGE questions count as in-domain
AD_PATTERN = os.environ.get("AD_PATTERN",
                            r"alzheimer|dementia|amyloid|tau protein|neurodegenerat")

QUERIES = [q.strip() for q in os.environ.get("QUERIES", "|".join([
    "Alzheimer's disease pathophysiology",
    "Alzheimer's disease treatment drugs",
    "amyloid beta plaques Alzheimer",
    "tau protein neurofibrillary tangles",
    "Alzheimer's disease risk factors genetics",
    "APOE4 Alzheimer's disease",
    "Alzheimer's disease biomarkers diagnosis",
    "dementia cognitive decline progression",
    "neuroinflammation Alzheimer's disease",
    "Alzheimer's disease clinical trials outcomes",
])).split("|") if q.strip()]

if not HF_TOKEN:
    raise SystemExit("[fatal] HF_TOKEN missing — pass --secrets HF_TOKEN")

# dtype: T4/P100 are pre-Ampere and do NOT support bf16
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    DTYPE = torch.bfloat16 if major >= 8 else torch.float16
    print(f"[setup] GPU {torch.cuda.get_device_name(0)} (sm_{major}{minor}) -> {DTYPE}", flush=True)
else:
    DTYPE = torch.float32
    print("[setup] WARNING no GPU detected — this will be unusably slow", flush=True)

WORK = os.environ.get("WORK_DIR", ".")
os.makedirs(WORK, exist_ok=True)
GRAPH_PATH = os.path.join(WORK, "alzheimers_kg.pkl")
EDGES_PATH = os.path.join(WORK, "alzheimers_edges.jsonl")
SEEN_PATH  = os.path.join(WORK, "seen_pmids.json")
QUES_PATH  = os.path.join(WORK, "ad_questions.json")
print(f"[setup] {DISEASE} | {N_ABSTRACTS} abstracts | repo {KG_REPO}", flush=True)

# ----------------------------- pull existing graph --------------------------
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=HF_TOKEN)
api.create_repo(KG_REPO, repo_type="dataset", exist_ok=True)

if FRESH_START:
    print("[fresh] FRESH_START=1 — ignoring any existing graph, rebuilding from zero", flush=True)
else:
    for local in (GRAPH_PATH, SEEN_PATH):
        try:
            p = hf_hub_download(repo_id=KG_REPO, repo_type="dataset",
                                filename=os.path.basename(local), token=HF_TOKEN)
            with open(p, "rb") as src, open(local, "wb") as dst:
                dst.write(src.read())
            print(f"[resume] pulled {os.path.basename(local)}", flush=True)
        except Exception:
            print(f"[resume] no existing {os.path.basename(local)} — fresh start", flush=True)

G = nx.MultiDiGraph()
if os.path.exists(GRAPH_PATH):
    try:
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)["graph"]
        print(f"[resume] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    except Exception as e:
        print(f"[resume] could not load graph ({type(e).__name__}) — starting fresh", flush=True)

SEEN = set()
if os.path.exists(SEEN_PATH):
    try:
        SEEN = set(json.load(open(SEEN_PATH)))
        print(f"[resume] {len(SEEN)} PMIDs already processed", flush=True)
    except Exception:
        pass

# ----------------------------- MIRAGE AD questions --------------------------
MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
urllib.request.urlretrieve(MIRAGE_URL, os.path.join(WORK, "mirage.json"))
_raw = json.load(open(os.path.join(WORK, "mirage.json")))

AD_PAT = re.compile(AD_PATTERN, re.I)
AD_QUESTIONS = []
for b, items in _raw.items():
    for qid, it in items.items():
        blob = it["question"] + " " + " ".join(it.get("options", {}).values())
        if AD_PAT.search(blob):
            AD_QUESTIONS.append({"benchmark": b, "id": qid, "question": it["question"],
                                 "options": it.get("options", {}),
                                 "answer": it.get("answer", "")})
json.dump(AD_QUESTIONS, open(QUES_PATH, "w"), indent=2)
print(f"[data] {len(AD_QUESTIONS)} in-domain questions: "
      f"{dict(Counter(q['benchmark'] for q in AD_QUESTIONS))}", flush=True)

# ----------------------------- PubMed harvest -------------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = {"User-Agent": f"ad-kg/0.1 (mailto:{NCBI_EMAIL})"}
SLEEP = 0.11 if NCBI_API_KEY else 0.34

def _get(url, params, is_json=True, retries=3):
    q = url + "?" + urllib.parse.urlencode(params)
    for a in range(retries):
        try:
            req = urllib.request.Request(q, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                t = r.read().decode("utf-8", "replace")
            return json.loads(t) if is_json else t
        except Exception as e:
            if a == retries - 1:
                print(f"  [pubmed] gave up: {type(e).__name__}", flush=True)
                return None
            time.sleep(1.5 * (a + 1))

def harvest(queries, target):
    per_q = max(target // max(len(queries), 1) + 5, 10)
    common = {"db": "pubmed", "retmode": "json", "email": NCBI_EMAIL}
    if NCBI_API_KEY: common["api_key"] = NCBI_API_KEY
    pmids = []
    for q in queries:
        time.sleep(SLEEP)
        j = _get(f"{EUTILS}/esearch.fcgi",
                 {**common, "term": q, "retmax": per_q, "sort": "relevance"})
        ids = (j or {}).get("esearchresult", {}).get("idlist", []) if j else []
        new = [i for i in ids if i not in SEEN]
        pmids += new
        print(f"  '{q[:42]}': {len(ids)} hits, {len(new)} new", flush=True)
    pmids = list(dict.fromkeys(pmids))[:target]

    pairs = []
    for i in range(0, len(pmids), 20):
        batch = pmids[i:i+20]
        time.sleep(SLEEP)
        txt = _get(f"{EUTILS}/efetch.fcgi",
                   {**{k: v for k, v in common.items() if k != "retmode"},
                    "id": ",".join(batch), "rettype": "abstract", "retmode": "text"},
                   is_json=False)
        if not txt: continue
        chunks = [c.strip() for c in txt.split("\n\n\n") if len(c.strip()) > 300]
        for j, c in enumerate(chunks):
            pmid = batch[j] if j < len(batch) else batch[-1]
            pairs.append((pmid, c[:3000]))
        if len(pairs) >= target:          # split can yield more chunks than PMIDs
            break
    return pairs[:target]

print(f"[harvest] up to {N_ABSTRACTS} abstracts ...", flush=True)
PAIRS = harvest(QUERIES, N_ABSTRACTS)
print(f"[harvest] {len(PAIRS)} abstracts retrieved", flush=True)
if not PAIRS:
    raise SystemExit("[fatal] no abstracts retrieved — check network / NCBI_EMAIL")

# ----------------------------- model ----------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
print(f"[gen] loading {GENERATOR} (~8.6 GB) ...", flush=True)
tok = AutoTokenizer.from_pretrained(GENERATOR, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    GENERATOR, dtype=DTYPE, device_map="auto", token=HF_TOKEN)
print(f"[gen] loaded on {model.device}", flush=True)

def gen(prompt, max_new=None, _diag=False):
    if max_new is None:
        max_new = MAX_NEW_TOK
    msgs = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    ilen = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    seq = out[0]
    # generate() may return prompt+completion OR completion only — handle both,
    # otherwise slicing at ilen silently yields an empty string.
    if seq.shape[0] > ilen:
        new_tokens = seq[ilen:]
    else:
        new_tokens = seq
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if _diag:
        full = tok.decode(seq, skip_special_tokens=True)
        print(f"[diag] prompt_tokens={ilen} out_tokens={seq.shape[0]} "
              f"new_tokens={new_tokens.shape[0]} decoded_chars={len(text)}", flush=True)
        if not text:
            print(f"[diag] sliced decode empty — full decode was {len(full)} chars; "
                  f"tail: {full[-200:]!r}", flush=True)
    return text

# ---- THROUGHPUT CANARY: measure real speed before committing to the full run
_t = time.time()
_probe = gen("List three risk factors for Alzheimer's disease.", max_new=64, _diag=True)
_el = time.time() - _t
_tps = 64 / _el
print(f"[canary] 64 tokens in {_el:.1f}s -> ~{_tps:.1f} tok/s", flush=True)
print(f"[canary] sample output: {_probe[:150]!r}", flush=True)
if not _probe:
    print("[canary] FATAL: model returned an empty string on a trivial prompt. "
          "Extraction cannot work — stop and investigate before burning GPU time.", flush=True)
_per_abs = (MAX_NEW_TOK / max(_tps, 0.1)) * 1.4      # x1.4 for occasional retries
print(f"[canary] est. {_per_abs:.0f}s per abstract -> "
      f"{_per_abs*len(PAIRS)/3600:.1f}h for {len(PAIRS)} abstracts", flush=True)
if _tps < 8:
    print("[canary] WARNING very slow — model may be partly on CPU. "
          "Consider --flavor a10g-small, or lower MAX_NEW_TOKENS.", flush=True)

def extract_json(text):
    # 1. try a complete array/object
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if m:
        for cand in (m.group(1), re.sub(r",\s*([}\]])", r"\1", m.group(1))):
            try:
                return json.loads(cand)
            except Exception:
                pass
    # 2. SALVAGE: pull individual {...} objects out of a truncated/unclosed array.
    #    Generation often stops mid-array; the objects before the cut are still good.
    objs = []
    for om in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        frag = om.group(0)
        for cand in (frag, re.sub(r",\s*\}", "}", frag)):
            try:
                objs.append(json.loads(cand))
                break
            except Exception:
                continue
    return objs if objs else None

TRIPLE_PROMPT = (
    "Extract medical relationships from this abstract about {disease}.\n"
    "Return ONLY a JSON array of objects:\n"
    '[{{"source":"<entity>","relation":"<short verb phrase>","target":"<entity>",'
    '"confidence":<1-10>}}]\n'
    "Rules: at most 6 relationships; only ones the text explicitly supports; "
    "use concise entity names (diseases, genes, proteins, drugs, biomarkers, "
    "risk factors); confidence reflects how strongly the abstract supports it. "
    "Return [] if none.\n\nAbstract:\n{abstract}\n\nJSON:")

_DEBUG_SHOWN = 0

def extract_triples(abstract, retries=1):
    global _DEBUG_SHOWN
    prompt = TRIPLE_PROMPT.format(disease=DISEASE, abstract=abstract[:2500])
    for attempt in range(retries + 1):
        raw = gen(prompt, _diag=(_DEBUG_SHOWN < 2))
        if _DEBUG_SHOWN < 2:          # show what the model actually emits, twice
            _DEBUG_SHOWN += 1
            print(f"[debug] raw model output (attempt {attempt+1}):\n"
                  f"{raw[:400]}\n[debug] ---", flush=True)
        js = extract_json(raw)
        if isinstance(js, list):
            out = []
            for t in js:
                if not isinstance(t, dict): continue
                s = str(t.get("source", "")).strip()
                r = str(t.get("relation", "")).strip()
                o = str(t.get("target", "")).strip()
                if not (s and r and o): continue
                try: c = int(t.get("confidence", 5))
                except Exception: c = 5
                out.append({"source": s, "relation": r, "target": o,
                            "confidence": max(1, min(10, c))})
            if out or attempt == retries:
                return out
    return []

# ----------------------------- graph merge ----------------------------------
# ----------------------------- entity canonicalisation ----------------------
# Without this, "APOE4", "APOE-epsilon 4 allele" and "apolipoprotein E4" become
# THREE separate nodes, splitting their edges and killing bridge-fact hits.
ALIASES = {
    "alzheimer's disease": ["alzheimer disease", "alzheimers disease", "alzheimer's",
                            "alzheimers", "alzheimer", "ad", "alzheimer's dementia",
                            "alzheimer type dementia", "sporadic alzheimer's disease",
                            "late-onset alzheimer's disease", "load"],
    "apoe4": ["apoe-4", "apoe e4", "apoe epsilon 4", "apoe epsilon 4 allele",
              "apoe-epsilon 4 allele", "apoe ε4", "apoe ε4 allele", "apolipoprotein e4",
              "apolipoprotein e epsilon 4", "apoe4 allele", "apoe e4 allele", "ε4 allele"],
    "amyloid beta": ["amyloid-beta", "abeta", "aβ", "amyloid β", "amyloid-β",
                     "beta-amyloid", "β-amyloid", "amyloid beta peptide",
                     "amyloid beta protein", "aβ peptide", "amyloid beta 42", "aβ42"],
    "amyloid plaques": ["amyloid plaque", "senile plaques", "senile plaque",
                        "beta-amyloid plaques", "amyloid deposits"],
    "neurofibrillary tangles": ["neurofibrillary tangle", "nfts", "nft",
                                "neurofibrillary degeneration"],
    "tau protein": ["tau", "microtubule-associated protein tau", "mapt protein"],
    "dementia": ["dementias", "cognitive impairment", "major neurocognitive disorder"],
}
_ALIAS = {}
for _canon, _vars in ALIASES.items():
    _ALIAS[_canon] = _canon
    for _v in _vars:
        _ALIAS[_v] = _canon

def norm(e):
    """Light normalisation: lowercase, collapse whitespace, strip stray punctuation."""
    s = re.sub(r"\s+", " ", e.strip().lower()).strip(" .,;:()[]")
    return re.sub(r"^(the|a|an)\s+", "", s)

def canonical(e):
    """Map an entity string to its canonical name via the alias table."""
    s = norm(e)
    if s in _ALIAS:
        return _ALIAS[s]
    flat = re.sub(r"\s+", " ", re.sub(r"[-_/']", " ", s)).strip()
    if flat in _ALIAS:
        return _ALIAS[flat]
    return s

def resolve(e, threshold=0.92):
    """Canonicalise, then fuzzy-match against existing nodes so spelling variants
    merge instead of fragmenting. Prefix-blocked so it stays fast as G grows."""
    c = canonical(e)
    if c in G:
        return c
    pre = c[:4]
    cands = [n for n in G.nodes() if n[:4] == pre]
    if cands:
        m = get_close_matches(c, cands, n=1, cutoff=threshold)
        if m:
            return m[0]
    return c

def add_triple(t, pmid=None):
    s, o = resolve(t["source"]), resolve(t["target"])
    if not s or not o or s == o: return False
    if t["confidence"] < CONF_MIN: return False
    for n, raw in ((s, t["source"]), (o, t["target"])):
        if n not in G:
            G.add_node(n, label=raw.strip())
    for _, d in (G.get_edge_data(s, o) or {}).items():
        if d.get("relation") == t["relation"]:
            d["confidence"] = max(d["confidence"], t["confidence"])
            d["n_evidence"] = d.get("n_evidence", 1) + 1
            return False
    G.add_edge(s, o, relation=t["relation"], confidence=t["confidence"],
               n_evidence=1, pmid=pmid)
    return True

def save_and_push(note=""):
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump({"graph": G, "disease": DISEASE}, f)
    with open(EDGES_PATH, "w") as f:
        for u, v, d in G.edges(data=True):
            f.write(json.dumps({"source": G.nodes[u].get("label", u),
                                "relation": d["relation"],
                                "target": G.nodes[v].get("label", v),
                                "confidence": d["confidence"],
                                "n_evidence": d.get("n_evidence", 1)}) + "\n")
    json.dump(sorted(SEEN), open(SEEN_PATH, "w"))
    try:
        for p in (GRAPH_PATH, EDGES_PATH, SEEN_PATH, QUES_PATH):
            api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                            repo_id=KG_REPO, repo_type="dataset")
        print(f"  [push] saved {note}", flush=True)
    except Exception as e:
        print(f"  [push] failed ({type(e).__name__}) — local copy kept", flush=True)

# ----------------------------- build ----------------------------------------
t0 = time.time(); new_edges = 0
try:
    for i, (pmid, abstract) in enumerate(PAIRS, 1):
        for t in extract_triples(abstract):
            if add_triple(t, pmid): new_edges += 1
        SEEN.add(pmid)
        if i % PRINT_EVERY == 0:
            rate = (time.time() - t0) / i
            print(f"[build] {i}/{len(PAIRS)} | {rate:.1f}s/abstract | "
                  f"ETA {rate*(len(PAIRS)-i)/60:.0f} min | "
                  f"graph {G.number_of_nodes()}n/{G.number_of_edges()}e", flush=True)
        if i % CHECKPOINT_N == 0:
            save_and_push(f"(checkpoint @ {i})")
finally:
    save_and_push("(final)")

print(f"\n=== GRAPH BUILT === nodes {G.number_of_nodes()} | edges {G.number_of_edges()} "
      f"| new {new_edges} | {(time.time()-t0)/60:.1f} min", flush=True)

# ----------------------------- coverage check -------------------------------
from difflib import get_close_matches  # (already imported above; kept for clarity)
NODE_NAMES = list(G.nodes())

def link(text, max_ents=6):
    tl = norm(text)
    hits = [n for n in NODE_NAMES if len(n) > 4 and n in tl]
    # also try alias table on the raw text so "APOE4" in a question finds the canonical node
    for alias, canon in _ALIAS.items():
        if len(alias) > 3 and alias in tl and canon in G:
            hits.append(canon)
    if not hits:
        for w in re.findall(r"[a-z]{5,}", tl)[:8]:
            hits += get_close_matches(w, NODE_NAMES, n=1, cutoff=0.9)
    return list(dict.fromkeys(hits))[:max_ents]

def facts_for(question, options, max_facts=8):
    q_ids = link(question)
    o_ids = link(" ".join(options.values())) if options else []
    bridge, neigh = [], []
    for a in q_ids:
        for b in o_ids:
            if a == b: continue
            for u, v in ((a, b), (b, a)):
                for _, d in (G.get_edge_data(u, v) or {}).items():
                    bridge.append((G.nodes[u].get("label", u), d["relation"],
                                   G.nodes[v].get("label", v), d["confidence"]))
    for a in q_ids:
        for nb_ in list(G.successors(a))[:6]:
            for _, d in (G.get_edge_data(a, nb_) or {}).items():
                neigh.append((G.nodes[a].get("label", a), d["relation"],
                              G.nodes[nb_].get("label", nb_), d["confidence"]))
    seen, out = set(), []
    for s, r, t_, c in bridge + neigh:
        k = (s, r, t_)
        if k in seen: continue
        seen.add(k); out.append(f"{s} —{r} (conf {c})→ {t_}")
        if len(out) >= max_facts: break
    return out, len(bridge) > 0, len(q_ids)

n_any = n_bridge = n_linked = 0
per_bench = {}
for q in AD_QUESTIONS:
    facts, has_bridge, n_ents = facts_for(q["question"], q.get("options", {}))
    b = q["benchmark"]
    per_bench.setdefault(b, {"n": 0, "facts": 0, "bridge": 0})
    per_bench[b]["n"] += 1
    if n_ents: n_linked += 1
    if facts: n_any += 1; per_bench[b]["facts"] += 1
    if has_bridge: n_bridge += 1; per_bench[b]["bridge"] += 1
    q["_facts"] = facts

N = max(len(AD_QUESTIONS), 1)
print("\n===== DOMAIN-KG COVERAGE on in-domain questions =====", flush=True)
print(f"entity-linked     : {n_linked}/{N} ({n_linked/N:.0%})", flush=True)
print(f"got >=1 fact      : {n_any}/{N} ({n_any/N:.0%})", flush=True)
print(f"got a BRIDGE fact : {n_bridge}/{N} ({n_bridge/N:.0%})   <-- Hetionet was 0%", flush=True)
for b, s in per_bench.items():
    print(f"  {b:<10} n={s['n']:<3} facts {s['facts']}/{s['n']}  bridge {s['bridge']}/{s['n']}", flush=True)

print("\n=== densest entities ===", flush=True)
for n, d in sorted(G.degree(), key=lambda x: -x[1])[:12]:
    print(f"  {G.nodes[n].get('label', n):<40} degree {d}", flush=True)

print("\n=== highest-confidence relations ===", flush=True)
for u, v, d in sorted(G.edges(data=True),
                      key=lambda e: (-e[2]["confidence"], -e[2].get("n_evidence", 1)))[:12]:
    print(f"  {G.nodes[u].get('label',u)} —{d['relation']} (conf {d['confidence']}, "
          f"{d.get('n_evidence',1)} papers)→ {G.nodes[v].get('label',v)}", flush=True)

shown = 0
for q in AD_QUESTIONS:
    if q.get("_facts") and shown < 3:
        print(f"\n[{q['benchmark']}] {q['question'][:100]}", flush=True)
        for f in q["_facts"][:4]: print("   -", f, flush=True)
        shown += 1

print(f"\n[done] graph at https://huggingface.co/datasets/{KG_REPO}", flush=True)