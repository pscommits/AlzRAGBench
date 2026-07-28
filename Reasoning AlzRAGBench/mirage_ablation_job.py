# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "sentence-transformers",
#     "faiss-cpu",
#     "networkx",
#     "datasets",
#     "numpy",
#     "huggingface_hub",
# ]
# ///
"""
Build the retrieval artifacts once and push them to a Hugging Face dataset repo,
so a Space can load them in seconds instead of rebuilding on every startup.

Saves:
  faiss.index        - FAISS index over the corpus embeddings
  corpus.jsonl       - the corpus passages, aligned 1:1 with the index rows
  hetionet.pkl       - pickled networkx graph + NAME2ID map
  meta.json          - encoder name, dims, counts (so the Space loads consistently)

Run:
    hf jobs uv run --flavor a10g-small --timeout 2h \
        --secrets HF_TOKEN \
        --env ARTIFACT_REPO=NathanPereira/kg-mmrag-artifacts \
        build_artifacts_job.py

Smoke test (tiny corpus):
    hf jobs uv run --flavor t4-small --timeout 40m \
        --secrets HF_TOKEN \
        --env ARTIFACT_REPO=NathanPereira/kg-mmrag-artifacts \
        --env CORPUS_MAX=3000 \
        build_artifacts_job.py
"""
import os, json, pickle, urllib.request, bz2
import numpy as np
import torch

CORPORA      = os.environ.get("CORPORA", "textbooks").split(",")
CORPUS_MAX   = int(os.environ.get("CORPUS_MAX", "0"))     # 0 = full corpus
ENCODER      = os.environ.get("ENCODER", "NeuML/pubmedbert-base-embeddings")
ARTIFACT_REPO = os.environ.get("ARTIFACT_REPO", "NathanPereira/kg-mmrag-artifacts")
BUILD_KG     = os.environ.get("BUILD_KG", "1") == "1"     # set 0 to skip Hetionet
HF_TOKEN     = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "artifacts"; os.makedirs(OUT, exist_ok=True)
print(f"[setup] device={DEVICE} corpora={CORPORA} corpus_max={CORPUS_MAX or 'full'}", flush=True)
if not ARTIFACT_REPO:
    print("[WARN] ARTIFACT_REPO is empty — artifacts will NOT be saved and the Job disk "
          "is wiped on exit. Set --env ARTIFACT_REPO=<user>/<name> to persist them.", flush=True)
elif not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — the push at the end will fail and the artifacts will "
          "be lost. Pass --secrets HF_TOKEN.", flush=True)
print(f"[setup] artifacts will be pushed to: {ARTIFACT_REPO or '(NONE — not saved)'}", flush=True)

# ---- 1 · corpus -------------------------------------------------------------
from datasets import load_dataset

def load_medrag_corpus(names, cap):
    passages = []
    for name in names:
        name = name.strip(); repo = f"MedRAG/{name}"
        print(f"[corpus] loading {repo} ...", flush=True)
        try:
            ds = load_dataset(repo, split="train")
            field = next((f for f in ("contents", "content", "text") if f in ds.column_names), None)
            if field is None:
                print(f"[corpus]   SKIP {repo}: no text field in {ds.column_names}", flush=True); continue
            for t in ds[field]:
                if t and len(t) > 20: passages.append(t)
            print(f"[corpus]   {repo}: {len(ds)} chunks", flush=True)
        except Exception as e:
            print(f"[corpus]   SKIP {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True); continue
    if not passages:
        raise RuntimeError("No corpus passages loaded from: " + ", ".join(names))
    if cap and len(passages) > cap:
        rng = np.random.default_rng(0)
        passages = [passages[i] for i in rng.choice(len(passages), cap, replace=False)]
    return passages

CORPUS = load_medrag_corpus(CORPORA, CORPUS_MAX)
print(f"[corpus] total {len(CORPUS)} passages", flush=True)

# ---- 2 · embed + FAISS index ------------------------------------------------
from sentence_transformers import SentenceTransformer
import faiss

enc = SentenceTransformer(ENCODER, device=DEVICE)
emb = enc.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=True,
                 batch_size=128, show_progress_bar=False)
dim = int(emb.shape[1])
index = faiss.IndexFlatIP(dim); index.add(emb)
faiss.write_index(index, f"{OUT}/faiss.index")
with open(f"{OUT}/corpus.jsonl", "w") as f:
    for p in CORPUS:
        f.write(json.dumps({"text": p}) + "\n")
print(f"[vector] indexed {index.ntotal} passages (dim {dim}) -> saved", flush=True)

# ---- 3 · Hetionet graph -----------------------------------------------------
if BUILD_KG:
    import networkx as nx
    HETIO = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
    urllib.request.urlretrieve(HETIO, "hetionet.json.bz2")
    with bz2.open("hetionet.json.bz2") as f:
        data = json.load(f)
    G = nx.MultiDiGraph()
    for n in data["nodes"]:
        nid = n["identifier"] if isinstance(n["identifier"], str) else str(n["identifier"])
        G.add_node(nid, name=n["name"], kind=n["kind"])
    NAME2ID = {d["name"].lower(): nid for nid, d in G.nodes(data=True)}
    for e in data["edges"]:
        G.add_edge(str(e["source_id"][1]), str(e["target_id"][1]), kind=e["kind"])
    with open(f"{OUT}/hetionet.pkl", "wb") as f:
        pickle.dump({"graph": G, "name2id": NAME2ID}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[kg] {G.number_of_nodes()} nodes {G.number_of_edges()} edges -> saved", flush=True)

# ---- 4 · meta + push --------------------------------------------------------
meta = {"encoder": ENCODER, "dim": dim, "n_passages": len(CORPUS),
        "corpora": CORPORA, "has_kg": BUILD_KG}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)

if ARTIFACT_REPO:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(ARTIFACT_REPO, repo_type="dataset", exist_ok=True)
    print(f"[push] uploading {OUT}/ to {ARTIFACT_REPO} ...", flush=True)
    api.upload_folder(folder_path=OUT, repo_id=ARTIFACT_REPO, repo_type="dataset")
    print(f"[push] done -> https://huggingface.co/datasets/{ARTIFACT_REPO}", flush=True)
else:
    print("[push] ARTIFACT_REPO unset — artifacts only on the ephemeral Job disk.", flush=True)