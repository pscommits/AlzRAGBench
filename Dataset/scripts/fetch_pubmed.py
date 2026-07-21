"""
Fetch 20 standard PubMed articles (abstract + metadata) on Alzheimer's disease,
one per core subtopic, via NCBI E-utilities. Saves JSON + plain-text files into
'Selected articles/'.
"""
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent.parent / "Selected articles"
OUT_DIR.mkdir(exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# One query per subtopic -> good topical spread across the corpus.
# Each maps to a short slug used for filenames / source IDs.
QUERIES = [
    ("pathology_amyloid", "Alzheimer's disease amyloid beta plaque pathology review"),
    ("pathology_tau", "Alzheimer's disease tau protein neurofibrillary tangles review"),
    ("genetics_apoe", "APOE4 Alzheimer's disease risk genetics review"),
    ("genetics_familial", "presenilin APP mutation familial Alzheimer's disease"),
    ("epidemiology", "Alzheimer's disease epidemiology prevalence incidence review"),
    ("diagnosis_criteria", "Alzheimer's disease diagnostic criteria clinical review"),
    ("biomarkers_csf", "cerebrospinal fluid biomarkers Alzheimer's disease amyloid tau review"),
    ("imaging_pet", "PET imaging amyloid tau Alzheimer's disease review"),
    ("imaging_mri", "MRI hippocampal atrophy Alzheimer's disease review"),
    ("mci", "mild cognitive impairment progression Alzheimer's disease review"),
    ("treatment_cholinesterase", "cholinesterase inhibitors donepezil Alzheimer's disease treatment review"),
    ("treatment_memantine", "memantine NMDA receptor Alzheimer's disease treatment"),
    ("treatment_immunotherapy", "anti-amyloid monoclonal antibody lecanemab aducanumab Alzheimer's disease"),
    ("neuroinflammation", "neuroinflammation microglia Alzheimer's disease review"),
    ("risk_factors_lifestyle", "modifiable risk factors lifestyle prevention Alzheimer's disease review"),
    ("risk_factors_vascular", "vascular risk factors cerebrovascular disease Alzheimer's dementia review"),
    ("synaptic_dysfunction", "synaptic dysfunction loss cognitive decline Alzheimer's disease"),
    ("sleep", "sleep disturbance glymphatic clearance Alzheimer's disease"),
    ("caregiving", "caregiver burden dementia Alzheimer's disease review"),
    ("sex_differences", "sex differences gender Alzheimer's disease risk review"),
]

HEADERS = {"User-Agent": "HybridRAG-dataset-builder/1.0 (research use; contact: prabhat0405005@gmail.com)"}


def esearch(term):
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": 5,
        "sort": "relevance",
        "retmode": "json",
    }
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parse_article(article_el):
    pmid = article_el.findtext(".//PMID", default="").strip()
    title = "".join(article_el.find(".//ArticleTitle").itertext()) if article_el.find(".//ArticleTitle") is not None else ""

    abstract_parts = []
    for ab_text in article_el.findall(".//Abstract/AbstractText"):
        label = ab_text.get("Label")
        text = "".join(ab_text.itertext())
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts).strip()

    authors = []
    for author in article_el.findall(".//AuthorList/Author"):
        last = author.findtext("LastName", default="")
        fore = author.findtext("ForeName", default="")
        if last:
            authors.append(f"{fore} {last}".strip())

    journal = article_el.findtext(".//Journal/Title", default="")
    year = (
        article_el.findtext(".//JournalIssue/PubDate/Year")
        or article_el.findtext(".//JournalIssue/PubDate/MedlineDate", default="")[:4]
    )

    mesh_terms = [
        mh.findtext("DescriptorName", default="")
        for mh in article_el.findall(".//MeshHeadingList/MeshHeading")
    ]
    mesh_terms = [m for m in mesh_terms if m]

    doi = ""
    for el_id in article_el.findall(".//ELocationID"):
        if el_id.get("EIdType") == "doi":
            doi = el_id.text
    if not doi:
        for aid in article_el.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text

    return {
        "pmid": pmid,
        "title": title.strip(),
        "abstract": abstract,
        "authors": authors,
        "journal": journal.strip(),
        "year": year,
        "mesh_terms": mesh_terms,
        "doi": doi or "",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def main():
    manifest = []
    seen_pmids = set()

    for slug, query in QUERIES:
        pmids = esearch(query)
        chosen = None
        for pmid in pmids:
            if pmid not in seen_pmids:
                chosen = pmid
                break
        if not chosen:
            print(f"[WARN] no unused result for {slug} ({query})")
            continue
        seen_pmids.add(chosen)

        xml_text = efetch([chosen])
        root = ET.fromstring(xml_text)
        article_el = root.find(".//PubmedArticle")
        if article_el is None:
            print(f"[WARN] could not parse article for {slug} / pmid {chosen}")
            continue

        record = parse_article(article_el)
        record["subtopic"] = slug
        record["source_type"] = "pubmed"
        record["article_id"] = f"pubmed_{record['pmid']}"

        if not record["abstract"]:
            print(f"[WARN] empty abstract for {slug} / pmid {chosen}, skipping")
            continue

        json_path = OUT_DIR / f"pubmed_{record['pmid']}_{slug}.json"
        txt_path = OUT_DIR / f"pubmed_{record['pmid']}_{slug}.txt"

        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        txt_content = (
            f"Title: {record['title']}\n"
            f"Authors: {', '.join(record['authors'])}\n"
            f"Journal: {record['journal']} ({record['year']})\n"
            f"PMID: {record['pmid']}\n"
            f"DOI: {record['doi']}\n"
            f"MeSH Terms: {', '.join(record['mesh_terms'])}\n"
            f"URL: {record['url']}\n\n"
            f"Abstract:\n{record['abstract']}\n"
        )
        txt_path.write_text(txt_content, encoding="utf-8")

        manifest.append({
            "article_id": record["article_id"],
            "subtopic": slug,
            "title": record["title"],
            "pmid": record["pmid"],
            "year": record["year"],
        })
        print(f"[OK] {slug}: {record['title'][:80]} (PMID {record['pmid']})")
        time.sleep(0.4)  # be polite to NCBI (no API key -> ~3 req/sec limit)

    manifest_path = OUT_DIR / "_pubmed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFetched {len(manifest)}/{len(QUERIES)} PubMed articles.")


if __name__ == "__main__":
    main()
