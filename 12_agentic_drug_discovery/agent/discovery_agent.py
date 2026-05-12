"""
discovery_agent.py — Autonomous Drug Discovery Agent (ReAct Loop)
==================================================================
Implements a ReAct (Reason + Act) agent that autonomously runs a
multi-step drug discovery campaign against a molecular target.

Architecture
------------
  AgentConfig       — hyperparameters for one campaign run
  MockLLM           — deterministic 20-step decision tree (no API needed)
  DrugDiscoveryAgent — main ReAct loop: think → act → observe → update
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.memory import AgentMemory, MoleculeEntry
from agent.tools import TOOLS, execute_tool, get_tool_descriptions


# ---------------------------------------------------------------------------
# 1. AGENT CONFIG
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    target: str = "CDK4/6"
    target_pIC50: float = 8.0           # success threshold
    max_steps: int = 20
    mock_llm: bool = True
    seed_smiles: str = "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)CCN1"
    backtrack_threshold: int = 5        # steps without improvement → backtrack
    convergence_threshold: float = 0.02 # pIC50 delta below which we converge
    verbose: bool = True


# ---------------------------------------------------------------------------
# 2. MOCK LLM — deterministic 20-step decision tree
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Rule-based LLM simulator. Returns a realistic (thought, action, action_input)
    triple for each step without any API calls.

    Step schedule
    -------------
    1  : literature_search — target biology
    2  : literature_search — hERG / DILI safety
    3  : similarity_search — seed molecule
    4  : property_calculator — seed molecule
    5  : admet_predictor — seed molecule
    6  : binding_affinity — seed molecule
    7  : similarity_search — best hit so far
    8  : property_calculator — best hit
    9  : binding_affinity — best hit
    10 : scaffold_hopper — best hit (3 hops)
    11 : binding_affinity — hop 1
    12 : binding_affinity — hop 2
    13 : binding_affinity — hop 3
    14 : lead_optimizer — best compound (potency)
    15 : property_calculator — optimised compound
    16 : admet_predictor — optimised compound
    17 : binding_affinity — optimised compound
    18 : lead_optimizer — best compound (solubility)
    19 : binding_affinity — solubility-optimised compound
    20 : visualize — top compounds
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._hop_smiles: List[str] = []
        self._opt_smiles: List[str] = []

    # ------------------------------------------------------------------
    def think(self, step: int, memory: AgentMemory, last_obs: str) -> Dict:
        """Return {"thought": str, "action": str, "action_input": dict}."""
        best = memory.current_best
        best_smi = best.canonical_smiles if best else self.config.seed_smiles
        best_score = best.binding_score if best and best.binding_score else 0.0

        # Parse hop SMILES from last observation if available
        if step in (11, 12, 13) and "hops" in last_obs:
            self._parse_hops(last_obs)

        # Parse optimised SMILES from lead_optimizer observations
        if step in (15, 16, 17) and "suggestions" in last_obs:
            self._parse_opt(last_obs)

        # ── Step schedule ──────────────────────────────────────────────
        if step == 1:
            return {
                "thought": (
                    f"Starting a discovery campaign against {self.config.target}. "
                    "I should first understand the target biology and key pharmacophore "
                    "requirements before exploring chemical space."
                ),
                "action": "literature_search",
                "action_input": {"query": self.config.target},
            }

        elif step == 2:
            return {
                "thought": (
                    "Good background on the target. Now I need to understand the "
                    "safety landscape — hERG cardiotoxicity and DILI are the most "
                    "common liabilities for this chemotype."
                ),
                "action": "literature_search",
                "action_input": {"query": "hERG DILI safety"},
            }

        elif step == 3:
            return {
                "thought": (
                    f"I have the safety context. Let me search for compounds similar "
                    f"to the seed molecule to understand what active scaffolds look like."
                ),
                "action": "similarity_search",
                "action_input": {"smiles": self.config.seed_smiles, "top_k": 5},
            }

        elif step == 4:
            return {
                "thought": (
                    "Similarity search returned hits. Let me calculate the physicochemical "
                    "properties of the seed to establish a baseline before scoring binding."
                ),
                "action": "property_calculator",
                "action_input": {"smiles": self.config.seed_smiles},
            }

        elif step == 5:
            return {
                "thought": (
                    "Properties look reasonable. I should check the ADMET profile of "
                    "the seed before investing in further optimisation — no point "
                    "improving potency if the compound has a fatal ADMET flaw."
                ),
                "action": "admet_predictor",
                "action_input": {"smiles": self.config.seed_smiles},
            }

        elif step == 6:
            return {
                "thought": (
                    "ADMET looks acceptable. Now let me get the predicted binding "
                    f"affinity for the seed against {self.config.target}."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": self.config.seed_smiles},
            }

        elif step == 7:
            return {
                "thought": (
                    f"Seed pIC50 is around {best_score:.2f}. The similarity search "
                    "earlier found some hits — let me search around the best one "
                    "to find higher-affinity analogues."
                ),
                "action": "similarity_search",
                "action_input": {"smiles": best_smi, "top_k": 5},
            }

        elif step == 8:
            return {
                "thought": (
                    "Found similar compounds. Let me calculate properties for the "
                    "best one to confirm it meets drug-likeness criteria."
                ),
                "action": "property_calculator",
                "action_input": {"smiles": best_smi},
            }

        elif step == 9:
            return {
                "thought": (
                    "Properties are within acceptable range. Scoring binding affinity "
                    "for the best compound found so far."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": best_smi},
            }

        elif step == 10:
            return {
                "thought": (
                    f"Current best pIC50 = {best_score:.2f}. "
                    f"{'Above target — exploring scaffold diversity.' if best_score >= self.config.target_pIC50 else 'Below target — trying scaffold hops to escape the current SAR.'} "
                    "Bioisostere replacements may improve potency or selectivity."
                ),
                "action": "scaffold_hopper",
                "action_input": {"smiles": best_smi, "n_hops": 3},
            }

        elif step == 11:
            hop = self._hop_smiles[0] if self._hop_smiles else best_smi
            return {
                "thought": (
                    "Got scaffold hops. Scoring the first hop — a ring replacement "
                    "that may improve binding through a different binding mode."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": hop},
            }

        elif step == 12:
            hop = self._hop_smiles[1] if len(self._hop_smiles) > 1 else best_smi
            return {
                "thought": (
                    "Scoring the second scaffold hop — an isosteric replacement "
                    "that retains the pharmacophore while changing the ring system."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": hop},
            }

        elif step == 13:
            hop = self._hop_smiles[2] if len(self._hop_smiles) > 2 else best_smi
            return {
                "thought": (
                    "Scoring the third scaffold hop. After this I will choose the "
                    "best compound and move to lead optimisation."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": hop},
            }

        elif step == 14:
            return {
                "thought": (
                    f"Best compound so far: pIC50={best_score:.2f}. "
                    "Applying lead optimisation focused on potency — adding "
                    "hydrogen bond donors at the hinge-binding region and "
                    "fluorine for metabolic stability."
                ),
                "action": "lead_optimizer",
                "action_input": {"smiles": best_smi, "optimize_for": "potency"},
            }

        elif step == 15:
            opt = self._opt_smiles[0] if self._opt_smiles else best_smi
            return {
                "thought": (
                    "Got potency-optimised suggestions. Calculating properties "
                    "to make sure the modifications haven't broken drug-likeness."
                ),
                "action": "property_calculator",
                "action_input": {"smiles": opt},
            }

        elif step == 16:
            opt = self._opt_smiles[0] if self._opt_smiles else best_smi
            return {
                "thought": (
                    "Properties still look good. Running ADMET check on the "
                    "optimised compound before scoring binding."
                ),
                "action": "admet_predictor",
                "action_input": {"smiles": opt},
            }

        elif step == 17:
            opt = self._opt_smiles[0] if self._opt_smiles else best_smi
            return {
                "thought": (
                    "ADMET is acceptable. Scoring binding affinity for the "
                    "potency-optimised compound."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": opt},
            }

        elif step == 18:
            return {
                "thought": (
                    f"pIC50 now {best_score:.2f}. "
                    "Solubility may be limiting — applying lead optimisation "
                    "for solubility to improve oral bioavailability."
                ),
                "action": "lead_optimizer",
                "action_input": {"smiles": best_smi, "optimize_for": "solubility"},
            }

        elif step == 19:
            opt = self._opt_smiles[-1] if self._opt_smiles else best_smi
            return {
                "thought": (
                    "Solubility-optimised compound generated. Scoring its "
                    "binding affinity — hoping the solubility gain didn't cost potency."
                ),
                "action": "binding_affinity",
                "action_input": {"smiles": opt},
            }

        else:  # step == 20 or beyond
            top = memory.molecules.get_top_k(k=5)
            top_smiles = [e.canonical_smiles for e in top] if top else [best_smi]
            return {
                "thought": (
                    f"Campaign nearing completion. Best pIC50 = {best_score:.2f}. "
                    "Visualising the top candidates for the final report."
                ),
                "action": "visualize",
                "action_input": {
                    "smiles_list": top_smiles[:5],
                    "title": f"{self.config.target}_top_candidates",
                },
            }

    # ------------------------------------------------------------------
    def _parse_hops(self, obs: str) -> None:
        """Extract hop SMILES from scaffold_hopper observation string."""
        import json, re
        # Try to pull SMILES-like tokens from observation text
        pattern = r"'smiles':\s*'([^']+)'"
        matches = re.findall(pattern, obs)
        if matches:
            self._hop_smiles = matches[:3]

    def _parse_opt(self, obs: str) -> None:
        """Extract optimised SMILES from lead_optimizer observation string."""
        import re
        pattern = r"'new_smiles':\s*'([^']+)'"
        matches = re.findall(pattern, obs)
        if matches:
            self._opt_smiles = matches[:2]


# ---------------------------------------------------------------------------
# 3. DRUG DISCOVERY AGENT
# ---------------------------------------------------------------------------

class DrugDiscoveryAgent:
    """
    ReAct agent that autonomously discovers drug candidates.

    Loop per step
    -------------
    1. Think  — MockLLM generates (thought, action, action_input)
    2. Act    — execute_tool dispatches to the relevant Tool
    3. Observe — format tool output as observation string
    4. Update — extract molecules, add to memory, update best
    5. Check  — convergence or backtrack condition
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.memory = AgentMemory()
        self.llm = MockLLM(self.config)
        self._last_obs = ""

    # ------------------------------------------------------------------
    def run(self) -> AgentMemory:
        """Execute the full ReAct campaign and return populated memory."""
        if self.config.verbose:
            print(f"\nTools available: {list(TOOLS.keys())}\n")

        for step in range(1, self.config.max_steps + 1):
            # ── 1. Think ──────────────────────────────────────────────
            decision = self.llm.think(step, self.memory, self._last_obs)
            thought = decision["thought"]
            action = decision["action"]
            action_input = decision["action_input"]

            if self.config.verbose:
                print(f"[Step {step:02d}] {action}({_fmt_input(action_input)})")
                print(f"  Thought: {thought[:100]}{'…' if len(thought) > 100 else ''}")

            # ── 2. Act ────────────────────────────────────────────────
            try:
                result = execute_tool(action, **action_input)
            except Exception as exc:
                result = {"error": str(exc)}

            # ── 3. Observe ────────────────────────────────────────────
            obs = _format_observation(action, result)
            self._last_obs = obs

            if self.config.verbose:
                print(f"  Obs   : {obs[:120]}{'…' if len(obs) > 120 else ''}")

            # ── 4. Update memory ──────────────────────────────────────
            self._process_observation(action, result, step)
            self.memory.add_trace(step, thought, action, action_input, obs)

            # ── 5. Convergence check ──────────────────────────────────
            if self._check_convergence(step):
                if self.config.verbose:
                    print(f"\n  [Converged at step {step}]")
                break

            if self.config.verbose:
                best = self.memory.current_best
                if best and best.binding_score:
                    print(f"  Best  : {best.name} pIC50={best.binding_score:.2f}")
                print()

        return self.memory

    # ------------------------------------------------------------------
    def _process_observation(self, action: str, result: Dict, step: int) -> None:
        """Extract molecules from tool results and add to memory."""
        if "error" in result:
            return

        from rdkit import Chem

        if action == "similarity_search":
            for hit in result.get("hits", []):
                smi = hit.get("smiles", "")
                entry = MoleculeEntry.from_smiles(
                    smi,
                    name=hit.get("name", f"hit_{step}"),
                    binding_score=hit.get("pIC50"),
                    source="similarity_search",
                    step_discovered=step,
                    properties={"similarity": hit.get("similarity", 0.0)},
                )
                if entry:
                    self.memory.molecules.add(entry)
                    self.memory.update_best(entry)
                    # Form hypothesis from top hit
                    if hit.get("similarity", 0) > 0.6:
                        self.memory.add_hypothesis(
                            f"Compound {entry.name} (Tanimoto={hit['similarity']:.2f}) "
                            f"shares key pharmacophore with active template."
                        )

        elif action == "binding_affinity":
            pred_pic50 = result.get("predicted_pIC50")
            if pred_pic50 is None:
                return
            # Find or create entry for the scored SMILES
            # We tag the most recently added molecule that lacks a score
            entries = list(self.memory.molecules._entries.values())
            unscored = [e for e in entries if e.binding_score is None]
            if unscored:
                target_entry = sorted(unscored, key=lambda e: e.step_discovered)[-1]
                target_entry.binding_score = pred_pic50
                self.memory.update_best(target_entry)
            else:
                # Create a new entry from the last action_input
                # (reconstructed from trace)
                pass

            if pred_pic50 >= self.config.target_pIC50:
                self.memory.add_hypothesis(
                    f"pIC50={pred_pic50:.2f} exceeds target {self.config.target_pIC50}. "
                    "Focus on metabolic stability and selectivity."
                )

        elif action == "scaffold_hopper":
            for hop in result.get("hops", []):
                if not hop.get("valid", False):
                    continue
                smi = hop.get("smiles", "")
                entry = MoleculeEntry.from_smiles(
                    smi,
                    name=f"hop_{step}_{hop.get('transformation','?')[:8]}",
                    source="scaffold_hop",
                    step_discovered=step,
                )
                if entry:
                    self.memory.molecules.add(entry)
                    self.memory.visited_scaffolds.add(
                        hop.get("transformation", "unknown")
                    )

        elif action == "lead_optimizer":
            for sug in result.get("suggestions", []):
                smi = sug.get("new_smiles", "")
                if not smi:
                    continue
                entry = MoleculeEntry.from_smiles(
                    smi,
                    name=f"opt_{step}",
                    source="optimization",
                    step_discovered=step,
                    annotations=[sug.get("rationale", "")],
                )
                if entry:
                    self.memory.molecules.add(entry)
                    self.memory.add_hypothesis(
                        f"Modification '{sug.get('modification')}': "
                        f"{sug.get('rationale', '')}"
                    )

        elif action == "admet_predictor":
            admet = dict(result)
            # Tag the most recently discovered molecule
            entries = list(self.memory.molecules._entries.values())
            if entries:
                latest = sorted(entries, key=lambda e: e.step_discovered)[-1]
                latest.admet_profile = admet
                if admet.get("hERG_risk") == "high":
                    self.memory.add_hypothesis(
                        "hERG liability detected — consider reducing basicity or aromaticity."
                    )

    # ------------------------------------------------------------------
    def _check_convergence(self, step: int) -> bool:
        """Return True if the campaign should stop early."""
        # Enough steps and hit target
        best = self.memory.current_best
        if best and best.binding_score and best.binding_score >= self.config.target_pIC50:
            if step >= 12:
                return True
        # Too many steps without improvement
        if self.memory.should_backtrack(self.config.backtrack_threshold):
            if step >= 15:
                return True
        return False

    # ------------------------------------------------------------------
    def generate_report(self) -> str:
        """Generate a concise final discovery report."""
        top = self.memory.molecules.get_top_k(k=10)
        best = self.memory.current_best
        lines = [
            "╔" + "═" * 62 + "╗",
            f"║  DISCOVERY REPORT — {self.config.target:<40} ║",
            "╠" + "═" * 62 + "╣",
            f"║  Steps run        : {len(self.memory.trace):<40} ║",
            f"║  Molecules found  : {len(self.memory.molecules):<40} ║",
        ]
        if best and best.binding_score:
            lines += [
                f"║  Best compound    : {best.name:<40} ║",
                f"║  Best pIC50       : {best.binding_score:<40.2f} ║",
                f"║  Target reached   : {'YES ✓' if best.binding_score >= self.config.target_pIC50 else 'NO  ✗':<40} ║",
            ]
        lines += ["╠" + "═" * 62 + "╣", "║  TOP CANDIDATES                                               ║"]
        for i, e in enumerate(top[:5], 1):
            score = f"{e.binding_score:.2f}" if e.binding_score else " n/a"
            line = f"  {i}. {e.name:<20} pIC50={score}  [{e.source}]"
            lines.append(f"║{line:<62} ║")
        if self.memory.hypotheses:
            lines += ["╠" + "═" * 62 + "╣", "║  KEY INSIGHTS                                                 ║"]
            for h in self.memory.hypotheses[:3]:
                for chunk in _wrap(h, 60):
                    lines.append(f"║  {chunk:<60} ║")
        lines.append("╚" + "═" * 62 + "╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _fmt_input(d: Dict) -> str:
    parts = []
    for k, v in d.items():
        val = str(v)[:40] + "…" if len(str(v)) > 40 else str(v)
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _format_observation(action: str, result: Dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    if action == "similarity_search":
        hits = result.get("hits", [])
        if not hits:
            return "No similar compounds found."
        top = hits[0]
        return (f"Found {len(hits)} hits. Top: {top['name']} "
                f"pIC50={top.get('pIC50', '?')} sim={top.get('similarity', 0):.2f}. "
                f"Full: {str(hits)}")
    if action == "binding_affinity":
        pic50 = result.get("predicted_pIC50")
        pic50_str = f"{pic50:.2f}" if pic50 is not None else "?"
        return (f"predicted_pIC50={pic50_str}, "
                f"class={result.get('activity_class', '?')}, "
                f"confidence={result.get('confidence', '?')}")
    if action == "property_calculator":
        return (f"MW={result.get('MW', '?'):.1f}, LogP={result.get('LogP', '?'):.2f}, "
                f"QED={result.get('QED', '?'):.2f}, Lipinski={'pass' if result.get('Lipinski_pass') else 'fail'}")
    if action == "scaffold_hopper":
        hops = result.get("hops", [])
        valid = [h for h in hops if h.get("valid")]
        return (f"{result.get('n_valid', 0)} valid hops. "
                f"Transformations: {[h.get('transformation') for h in valid]}. "
                f"Full: {str(hops)}")
    if action == "lead_optimizer":
        sugs = result.get("suggestions", [])
        return (f"{len(sugs)} suggestions. "
                f"Full: {str(sugs)}")
    if action == "admet_predictor":
        return (f"BBB={result.get('BBB_penetrant')}, "
                f"hERG={result.get('hERG_risk')}, "
                f"CYP3A4={result.get('CYP3A4_substrate')}, "
                f"F={result.get('bioavailability_pct', 0):.0f}%")
    if action == "literature_search":
        results = result.get("results", [])
        if results:
            return results[0].get("summary", "")[:200]
        return "No results."
    if action == "visualize":
        return f"Saved to {result.get('saved_to', '?')} ({result.get('n_molecules', 0)} molecules)"
    return str(result)[:200]


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def action_input_smi(result: Dict) -> str:
    return ""


# ---------------------------------------------------------------------------
# QUICK DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = AgentConfig(target="CDK4/6", max_steps=20, verbose=True)
    agent = DrugDiscoveryAgent(config=config)
    memory = agent.run()
    print("\n" + agent.generate_report())
