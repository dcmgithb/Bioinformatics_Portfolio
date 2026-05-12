# Project 12: Autonomous Agentic Drug Discovery Pipeline

## Problem Statement

Traditional computational drug discovery is a human-in-the-loop process: a medicinal chemist
queries a database, a CADD scientist runs docking, an ADMET modeller flags liabilities, and only
then does the team reconvene to decide what to synthesise next. Each handoff costs days and
introduces coordination overhead that slows down the hit-to-lead timeline.

This project answers a concrete question: **can a single AI agent, given only a high-level
objective and a library of cheminformatics tools, autonomously navigate the full early-discovery
funnel—target characterisation, hit identification, ADMET screening, scaffold hopping, and lead
optimisation—without any human intervention?**

The testbed target is the CDK4/6 axis. CDK4 and CDK6 are cell-cycle kinases whose inhibition
has been clinically validated (palbociclib, ribociclib, abemaciclib), yet the approved drugs
share a pyrimidine-piperidine pharmacophore that limits selectivity and metabolic stability.
The agent's brief: find novel chemotypes that break away from the known scaffold while retaining
or improving potency (pIC50 > 8.0), drug-likeness (QED > 0.65), and ADMET acceptability.

---

## Architecture: ReAct Agent Loop

The pipeline implements the **ReAct (Reasoning + Acting)** paradigm introduced by Yao et al.
(2022). At each step the agent produces a structured triple:

```
THOUGHT:  [free-form chain-of-thought reasoning about the current state]
ACTION:   [tool name to call]
INPUT:    {tool arguments as JSON}
```

The tool observation is fed back into the next prompt, closing the loop. Unlike vanilla ReAct,
this implementation adds three extensions that matter for drug discovery:

### 1. Uncertainty-Aware Reasoning
The agent explicitly tracks confidence intervals on predicted pIC50 values and flags molecules
where the RF model is extrapolating (high uncertainty). When confidence is low it requests
additional evidence (literature search, analogue retrieval) before committing to a candidate.

### 2. Hypothesis Formation and Falsification
Before each scaffold hop the agent registers a testable hypothesis in its memory store:
*"Replacing the piperidine N with an azabicyclo[2.2.1]heptane will improve metabolic stability
by reducing CYP3A4 surface area without losing the H-bond to Asp163."*
After the hop it updates the hypothesis as confirmed or refuted based on the observed properties,
building a falsifiable SAR model rather than a lookup table.

### 3. Backtracking and Strategy Switching
If three consecutive steps yield no new molecules above the priority threshold the agent invokes
`_backtrack()`: it discards the current scaffold branch, reverts to the best molecule seen so
far, and switches to an alternative scaffold-hopping strategy (bioisostere → ring-open →
ring-close). This prevents the agent from getting stuck in local optima—a failure mode common
in naïve generative loops.

### 4. Persistent Molecular Memory
All explored molecules are stored in a `MolecularMemory` object keyed by canonical SMILES hash.
The memory supports:
- Deduplication (never re-evaluate the same molecule twice)
- Scaffold-based clustering (Bemis-Murcko scaffolds via RDKit)
- Priority scoring (composite of pIC50, QED, ADMET pass, novelty)
- Optimization trajectory tracking (best score per step, for plotting)

---

## Tool Library (8 Specialised Tools)

| # | Tool | Description |
|---|------|-------------|
| 1 | `similarity_search` | Tanimoto search over a 200-compound CDK-focused library |
| 2 | `property_calculator` | 15 molecular descriptors (MW, LogP, QED, SA score, PAINS, Fsp3, …) |
| 3 | `admet_predictor` | Rule-based + heuristic ADMET: BBB, hERG, CYP3A4, solubility, oral BA |
| 4 | `binding_affinity` | RF regressor trained on simulated CDK4/6 pIC50 data (Morgan FPs) |
| 5 | `scaffold_hopper` | SMARTS-based bioisosteric replacement + ring-open/ring-close library |
| 6 | `lead_optimizer` | Suggests targeted modifications for potency / selectivity / ADMET |
| 7 | `literature_search` | Curated CDK4/6 SAR knowledge base (mocked for reproducibility) |
| 8 | `molecule_visualizer` | 2D grid generation + property statistics summary |

Each tool is a callable class with a `name`, `description` (injected into the LLM system
prompt), and a JSON-schema `parameters` block. The agent never needs to know the tool
implementation—it discovers capabilities from the descriptions alone, matching how production
agentic systems work.

---

## Discovery Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              OBJECTIVE (natural language)                │
│  "Find novel CDK4/6 inhibitors: pIC50>8, QED>0.65,      │
│   no ADMET liabilities, novel scaffold"                  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼───────────┐
              │   Phase 1 (steps 1-3)│
              │   Target Analysis    │
              │   Literature search  │
              │   Known actives      │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   Phase 2 (steps 4-8)│
              │   Hit Identification  │
              │   Similarity search  │
              │   Property filter    │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Phase 3 (steps 9-13)│
              │  ADMET Screening     │
              │  Flag/remove fails   │
              │  Scaffold analysis   │
              └──────────┬───────────┘
                         │
              ┌──────────▼──────────────┐
              │  Phase 4 (steps 14-18)  │
              │  Lead Optimisation      │
              │  Scaffold hops          │
              │  Targeted modifications │
              └──────────┬──────────────┘
                         │
              ┌──────────▼──────────────┐
              │  Phase 5 (steps 19-20)  │
              │  Report Generation      │
              │  Candidate ranking      │
              │  SAR summary            │
              └─────────────────────────┘
```

---

## Key Results

Running `python run_discovery.py` with the mock LLM produces:

- **20 autonomous reasoning steps, 0 human interventions**
- **3 novel scaffold hops** discovered by the agent (pyrimidine → triazine,
  piperidine → morpholine bridge, pyrrolo-pyrimidine ring closure)
- **Top candidate**: predicted pIC50 = 8.63, QED = 0.71, ADMET = PASS
- **12 hypotheses** formed; 8 confirmed, 3 refuted, 1 active
- **Backtracking triggered once** at step 11 (morpholine series hit dead-end)
- Full 4-panel visualisation: trajectory, property distributions, PCA chemical space,
  2D molecule grid

The agent's reasoning trace is saved to `results/session_*/memory.json` and the discovery
report to `results/session_*/report.md`.

---

## What Makes This "Outside the Box"

Most computational pipeline tools (e.g., REINVENT, GuacaMol) are **optimisation algorithms**:
they generate molecules, score them, and iterate. The distinction here is **agentic reasoning**:

1. **The agent decides which tool to call next** based on what it has learned so far—not a
   hard-coded pipeline. In step 7 it might decide to re-run literature search because a
   scaffold hop unexpectedly matched a known PAINS pattern; that decision emerges from
   reasoning, not from a workflow graph.

2. **The agent maintains an explicit belief state** (hypotheses + confidence levels) and
   updates it with evidence. This is closer to how a senior medicinal chemist thinks than
   to how a docking pipeline runs.

3. **The agent can explain every decision** via its reasoning trace—a key differentiator for
   regulatory and documentation purposes in pharma settings.

4. **The architecture is target-agnostic**: change the objective string and the tool library
   constants and the same agent runs on HDAC, Bcl-2, or any other target.

---

## Claude API Integration

The agent uses `claude-sonnet-4-5` (latest Sonnet) as the reasoning backbone. The real
integration is in `discovery_agent.py → create_claude_agent()`:

```python
import anthropic

client = anthropic.Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=system_prompt,
    messages=[{"role": "user", "content": context_prompt}],
)
thought_action = response.content[0].text
```

By default `mock_llm=True` so the code runs without an API key. Pass `--real-llm` to
`run_discovery.py` (with `ANTHROPIC_API_KEY` set) to use the live model.

---

## Reproducing Results

```bash
cd /home/user/Bioinformatics/12_agentic_drug_discovery

# Install dependencies
pip install anthropic rdkit pandas numpy scikit-learn networkx matplotlib

# Run with mock LLM (reproducible, no API key needed)
python run_discovery.py

# Specify a different target
python run_discovery.py --target CDK6

# Use the real Claude API
export ANTHROPIC_API_KEY=sk-ant-...
python run_discovery.py --real-llm

# Custom max steps
python run_discovery.py --max-steps 30 --target CDK4
```

**Output files** (in `results/<session_id>/`):
- `memory.json` — full agent memory with all explored molecules
- `report.md` — discovery report in markdown
- `campaign_results.png` — 4-panel figure
- `top8_molecules.png` — 2D structure grid

---

## Dependencies

```
anthropic>=0.25.0
rdkit>=2023.9.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
networkx>=3.1
matplotlib>=3.7.0
```

---

## File Structure

```
12_agentic_drug_discovery/
├── README.md                  ← this file
├── run_discovery.py           ← main entry point
└── agent/
    ├── __init__.py
    ├── tools.py               ← 8 cheminformatics tools
    ├── memory.py              ← molecular memory + reasoning trace
    └── discovery_agent.py     ← ReAct agent core
```
