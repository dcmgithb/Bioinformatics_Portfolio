"""
memory.py — Agent Memory for Autonomous Drug Discovery
=======================================================
Tracks discovered molecules, reasoning traces, and hypotheses
across a multi-step ReAct discovery campaign.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem


# ---------------------------------------------------------------------------
# 1. MOLECULE ENTRY
# ---------------------------------------------------------------------------

@dataclass
class MoleculeEntry:
    """A single molecule discovered during the campaign."""
    smiles: str
    canonical_smiles: str
    name: str
    properties: Dict[str, float] = field(default_factory=dict)
    binding_score: Optional[float] = None          # predicted pIC50
    admet_profile: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"                         # similarity_search | scaffold_hop | optimization
    step_discovered: int = 0
    annotations: List[str] = field(default_factory=list)
    rank: int = 0

    @classmethod
    def from_smiles(cls, smiles: str, name: str = "", **kwargs) -> Optional["MoleculeEntry"]:
        """Create entry from SMILES; returns None if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        return cls(smiles=smiles, canonical_smiles=canonical,
                   name=name or smiles[:20], **kwargs)


# ---------------------------------------------------------------------------
# 2. REASONING TRACE
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """One step of the agent's ReAct loop."""
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: float = field(default_factory=time.time)

    def to_text(self) -> str:
        return (
            f"Step {self.step}\n"
            f"  Thought    : {self.thought}\n"
            f"  Action     : {self.action}({self.action_input})\n"
            f"  Observation: {self.observation[:120]}{'…' if len(self.observation) > 120 else ''}"
        )


# ---------------------------------------------------------------------------
# 3. MOLECULAR MEMORY
# ---------------------------------------------------------------------------

class MolecularMemory:
    """
    Stores unique molecules deduplicated by canonical SMILES.
    Supports top-k retrieval and source-based filtering.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._entries: Dict[str, MoleculeEntry] = {}   # canonical_smiles → entry
        self.max_size = max_size

    # ------------------------------------------------------------------
    def add(self, entry: MoleculeEntry) -> bool:
        """Add molecule. Returns True if new, False if duplicate."""
        key = entry.canonical_smiles
        if key in self._entries:
            # Update binding score if better
            existing = self._entries[key]
            if (entry.binding_score is not None and
                    (existing.binding_score is None or
                     entry.binding_score > existing.binding_score)):
                existing.binding_score = entry.binding_score
            return False
        if len(self._entries) >= self.max_size:
            return False
        self._entries[key] = entry
        return True

    # ------------------------------------------------------------------
    def get_top_k(self, k: int = 10, by: str = "binding_score") -> List[MoleculeEntry]:
        """Return top-k molecules sorted by field (descending)."""
        entries = list(self._entries.values())
        if by == "binding_score":
            scored = [e for e in entries if e.binding_score is not None]
            scored.sort(key=lambda e: e.binding_score, reverse=True)
            return scored[:k]
        # Fallback: sort by step_discovered (most recent)
        entries.sort(key=lambda e: e.step_discovered, reverse=True)
        return entries[:k]

    # ------------------------------------------------------------------
    def get_by_source(self, source: str) -> List[MoleculeEntry]:
        return [e for e in self._entries.values() if e.source == source]

    # ------------------------------------------------------------------
    def all_smiles(self) -> List[str]:
        return [e.canonical_smiles for e in self._entries.values()]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all entries to a ranked DataFrame."""
        rows = []
        for e in self._entries.values():
            row = {
                "name": e.name,
                "smiles": e.canonical_smiles,
                "binding_score": e.binding_score,
                "source": e.source,
                "step_discovered": e.step_discovered,
            }
            row.update(e.properties)
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "binding_score" in df.columns:
            df = df.sort_values("binding_score", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
        return df


# ---------------------------------------------------------------------------
# 4. AGENT MEMORY
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Combined memory store for a discovery campaign:
      - Molecular memory (deduplicated compound library)
      - Reasoning trace (full ReAct log)
      - Hypothesis tracker
      - Campaign statistics
    """

    def __init__(self) -> None:
        self.molecules = MolecularMemory()
        self.trace: List[ReasoningTrace] = []
        self.hypotheses: List[str] = []
        self.current_best: Optional[MoleculeEntry] = None
        self.steps_without_improvement: int = 0
        self.visited_scaffolds: set = set()
        self._best_score_history: List[float] = []

    # ------------------------------------------------------------------
    def add_trace(
        self,
        step: int,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: str,
    ) -> None:
        self.trace.append(ReasoningTrace(
            step=step,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        ))

    # ------------------------------------------------------------------
    def update_best(self, entry: MoleculeEntry) -> bool:
        """
        Update current best if entry has higher binding score.
        Returns True if a new best was set.
        """
        if entry.binding_score is None:
            return False
        if (self.current_best is None or
                entry.binding_score > (self.current_best.binding_score or -999)):
            self.current_best = entry
            self.steps_without_improvement = 0
            self._best_score_history.append(entry.binding_score)
            return True
        self.steps_without_improvement += 1
        return False

    # ------------------------------------------------------------------
    def add_hypothesis(self, hypothesis: str) -> None:
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)

    # ------------------------------------------------------------------
    def should_backtrack(self, threshold: int = 5) -> bool:
        """True if no improvement for `threshold` consecutive steps."""
        return self.steps_without_improvement >= threshold

    # ------------------------------------------------------------------
    def score_trajectory(self) -> List[float]:
        """Return best score at each step where a new best was found."""
        return list(self._best_score_history)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist memory to JSON."""
        data = {
            "molecules": [
                {
                    "smiles": e.canonical_smiles,
                    "name": e.name,
                    "binding_score": e.binding_score,
                    "source": e.source,
                    "step_discovered": e.step_discovered,
                    "properties": e.properties,
                    "annotations": e.annotations,
                }
                for e in self.molecules._entries.values()
            ],
            "trace": [
                {
                    "step": t.step,
                    "thought": t.thought,
                    "action": t.action,
                    "action_input": t.action_input,
                    "observation": t.observation,
                    "timestamp": t.timestamp,
                }
                for t in self.trace
            ],
            "hypotheses": self.hypotheses,
            "best_score_history": self._best_score_history,
            "current_best": self.current_best.canonical_smiles if self.current_best else None,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Restore memory from JSON."""
        with open(path) as f:
            data = json.load(f)
        for m in data.get("molecules", []):
            entry = MoleculeEntry.from_smiles(
                m["smiles"], name=m.get("name", ""),
                binding_score=m.get("binding_score"),
                source=m.get("source", "unknown"),
                step_discovered=m.get("step_discovered", 0),
                properties=m.get("properties", {}),
                annotations=m.get("annotations", []),
            )
            if entry:
                self.molecules.add(entry)
        self.hypotheses = data.get("hypotheses", [])
        self._best_score_history = data.get("best_score_history", [])

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Brief text summary of the campaign state."""
        n_mol = len(self.molecules)
        n_steps = len(self.trace)
        best = self.current_best
        best_str = (f"{best.name} (pIC50={best.binding_score:.2f})"
                    if best and best.binding_score else "none")
        sources = {}
        for e in self.molecules._entries.values():
            sources[e.source] = sources.get(e.source, 0) + 1
        src_str = ", ".join(f"{k}:{v}" for k, v in sources.items())
        lines = [
            "=" * 56,
            "  DISCOVERY CAMPAIGN MEMORY SUMMARY",
            "=" * 56,
            f"  Steps completed    : {n_steps}",
            f"  Molecules found    : {n_mol}",
            f"  Sources            : {src_str or 'none'}",
            f"  Current best       : {best_str}",
            f"  Hypotheses formed  : {len(self.hypotheses)}",
            f"  Steps w/o improve  : {self.steps_without_improvement}",
            "=" * 56,
        ]
        if self.hypotheses:
            lines.append("  Hypotheses:")
            for i, h in enumerate(self.hypotheses[:5], 1):
                lines.append(f"    {i}. {h}")
        return "\n".join(lines)
