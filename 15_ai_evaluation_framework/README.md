# AI Evaluation Framework for Precision Medicine

> Scientific accuracy rubrics · Biomedical fact-checking · Inter-annotator agreement ·
> Hallucination detection · Quality dashboard for AI-generated clinical content

---

## Motivation

As AI systems increasingly generate clinical and genomic content, rigorous evaluation
pipelines are essential to ensure scientific accuracy, clinical relevance, and regulatory
compliance before outputs are used in research or care settings. This framework provides
the tools to assess, annotate, and quality-control AI-generated precision medicine content
at scale — combining automated fact-checking with structured human annotation workflows.

```
AI Output
   │
   ├─[evaluation_rubric.py]──── Structured scoring (1–5 per dimension)
   │                             Scientific accuracy · Clinical relevance · Completeness
   │
   ├─[biomedical_factchecker.py] Entity extraction + KB verification
   │                             Gene/variant/drug claims · Hallucination flagging
   │
   ├─[annotation_pipeline.py]── Multi-annotator simulation + IAA
   │                             Cohen's κ · Krippendorff's α · Adjudication
   │
   └─[quality_dashboard.py]──── Aggregated quality metrics + report
                                 4-panel dashboard · Pass/fail per output
```

---

## Key Results (50 simulated AI outputs, 3 annotators)

```
Mean rubric score          3.6 / 5.0
Scientific accuracy        3.4 / 5.0  (most variable dimension)
Hallucination rate         18%  (gene/variant claim errors)
Inter-annotator κ          0.72  (substantial agreement)
Krippendorff's α           0.69  (ordinal scores)
Low-agreement items        12%  (flagged for adjudication)
```

---

## Modules

| Module | Method | Highlights |
|--------|--------|------------|
| `evaluation_rubric.py` | Weighted 6-dimension rubric | Calibration vs ground truth, score distribution |
| `biomedical_factchecker.py` | Entity extraction + KB lookup | Claim-level confidence, hallucination rate by topic |
| `annotation_pipeline.py` | Cohen's κ + Krippendorff's α | Adjudication workflow, drift detection |
| `quality_dashboard.py` | Aggregated 4-panel figure | Trend analysis, quality_report.txt |

---

## Dependencies

```
Python >= 3.10
numpy, pandas, scipy, scikit-learn
matplotlib, seaborn
```

## Quick Start

```bash
python evaluation_rubric.py         # score AI outputs on 6 dimensions
python biomedical_factchecker.py    # fact-check claims against KB
python annotation_pipeline.py      # IAA metrics + adjudication
python quality_dashboard.py         # full dashboard + quality report
```
