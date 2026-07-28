"""Knowledge graph retrieval — the relational half of the system.

Dense retrieval returns things that *look* alike. The KG returns things that
are *clinically related*: anatomy -> finding -> condition. That distinction is
the reason this module exists, and the reason fused retrieval beats dense-only.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Set

import networkx as nx

from src.utils import get_logger

logger = get_logger(__name__)


class KnowledgeGraphRetriever:
    def __init__(self, cfg, encoder=None):
        self.cfg = cfg
        self.encoder = encoder
        self.gcfg = cfg.retrieval.graph
        self.graph = nx.MultiDiGraph()
        self._entity_index: Dict[str, str] = {}  # normalised surface form -> node id

    # ---------------- build ----------------
    def build(self, triples: List[Dict[str, str]]) -> None:
        """triples: [{"head", "relation", "tail", "source"?}, ...]"""
        logger.info("Building KG from %d triples", len(triples))

        for t in triples:
            head, rel, tail = t["head"], t["relation"], t["tail"]
            for node in (head, tail):
                if node not in self.graph:
                    self.graph.add_node(node, label=node, type=t.get("type", "entity"))
            self.graph.add_edge(head, tail, relation=rel, source=t.get("source", "unknown"))

        self._build_entity_index()
        logger.info(
            "KG built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _build_entity_index(self) -> None:
        self._entity_index = {self._norm(n): n for n in self.graph.nodes()}

    @staticmethod
    def _norm(s: str) -> str:
        return s.lower().strip().replace("_", " ")

    # ---------------- entity linking ----------------
    def link_entities(self, text: str) -> List[str]:
        """Map free text -> KG node ids.

        `exact` is a deliberately dumb baseline. TODO(Nathan): the `embedding`
        linker is where the recall gains are — worth an ablation row.
        """
        mode = self.gcfg.entity_linker
        norm_text = self._norm(text)

        if mode == "exact":
            return [
                node for surface, node in self._entity_index.items() if surface in norm_text
            ]

        if mode == "fuzzy":
            from difflib import SequenceMatcher

            scored = [
                (node, SequenceMatcher(None, surface, norm_text).ratio())
                for surface, node in self._entity_index.items()
            ]
            scored.sort(key=lambda x: -x[1])
            return [n for n, s in scored[: self.gcfg.top_k_entities] if s > 0.6]

        if mode == "embedding":
            raise NotImplementedError(
                "TODO(Nathan): encode node labels with BiomedCLIP text encoder, "
                "cosine-match against the query."
            )

        raise ValueError(f"Unknown entity_linker: {mode}")

    # ---------------- traversal ----------------
    def get_subgraph(self, seed_nodes: List[str], max_hops: int | None = None) -> nx.MultiDiGraph:
        """N-hop neighbourhood around the seeds, capped so a hub node
        (e.g. 'lung') cannot drag in half the graph."""
        max_hops = max_hops or self.gcfg.max_hops
        cap = self.gcfg.max_subgraph_nodes
        allowed: Set[str] | None = set(self.gcfg.edge_types) if self.gcfg.edge_types else None

        visited: Set[str] = set(n for n in seed_nodes if n in self.graph)
        frontier = set(visited)

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                for _, nbr, data in self.graph.out_edges(node, data=True):
                    if allowed and data.get("relation") not in allowed:
                        continue
                    next_frontier.add(nbr)
                for nbr, _, data in self.graph.in_edges(node, data=True):
                    if allowed and data.get("relation") not in allowed:
                        continue
                    next_frontier.add(nbr)

            new = next_frontier - visited
            if not new:
                break
            visited |= new
            if len(visited) >= cap:
                logger.debug("Subgraph capped at %d nodes", cap)
                break
            frontier = new

        return self.graph.subgraph(list(visited)[:cap]).copy()

    def retrieve(self, query_text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Query text -> evidence triples, verbalised for the generator."""
        seeds = self.link_entities(query_text)[: self.gcfg.top_k_entities]
        if not seeds:
            logger.debug("No entities linked for query: %s", query_text[:60])
            return []

        sub = self.get_subgraph(seeds)
        results = []
        for head, tail, data in sub.edges(data=True):
            rel = data.get("relation", "related_to")
            results.append(
                {
                    "text": f"{head} {rel.replace('_', ' ')} {tail}",
                    "head": head,
                    "relation": rel,
                    "tail": tail,
                    "source": "graph",
                    # Seed-adjacent triples are more trustworthy than 2-hop ones.
                    "score": 1.0 if head in seeds or tail in seeds else 0.5,
                }
            )

        results.sort(key=lambda r: -r["score"])
        return results[: (top_k or self.gcfg.top_k_entities * 2)]

    # ---------------- persist ----------------
    def save(self, graph_dir: str | Path) -> None:
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        with (graph_dir / "kg.pkl").open("wb") as f:
            pickle.dump(self.graph, f)

        degrees = [d for _, d in self.graph.degree()]
        stats = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "relation_types": sorted(
                {d.get("relation") for _, _, d in self.graph.edges(data=True)}
            ),
            "avg_degree": round(sum(degrees) / len(degrees), 3) if degrees else 0,
        }
        with (graph_dir / "stats.json").open("w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved KG -> %s | %s", graph_dir, stats)

    def load(self, graph_dir: str | Path) -> None:
        graph_dir = Path(graph_dir)
        with (graph_dir / "kg.pkl").open("rb") as f:
            self.graph = pickle.load(f)
        self._build_entity_index()
        logger.info(
            "Loaded KG: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
