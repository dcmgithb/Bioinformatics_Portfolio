"""Synthetic seed data generator for the Antibody Discovery LIMS.

Generates realistic data for all 9 tables:
- 5 users, 4 instruments, 20 donors
- 3 samples per donor (60 total)
- 2 flow cytometry runs per sample (120 total)
- ~8 sequences per sample (~480 total)
- 1-2 expression results for top 100 sequences
- 1-3 assay results for top 200 sequences

Usage:
    python seed_data.py --db-url postgresql://user:pass@localhost:5432/lims_db
    python seed_data.py  # reads DATABASE_URL from environment
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import string
import sys
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Ensure models are importable when running from repo root or this directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models import (  # noqa: E402
    AntibodySequence,
    AssayResult,
    AuditLog,
    Base,
    BCellSample,
    Donor,
    ExpressionResult,
    FlowCytometryRun,
    Instrument,
    User,
)

# ---------------------------------------------------------------------------
# Constants — realistic biological data
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

VALID_AA: str = "ACDEFGHIKLMNPQRSTVWY"

VH_GENES: list[str] = [
    "IGHV1-2", "IGHV1-3", "IGHV1-8", "IGHV1-18", "IGHV1-24", "IGHV1-46", "IGHV1-69",
    "IGHV2-5", "IGHV2-26", "IGHV2-70",
    "IGHV3-7", "IGHV3-9", "IGHV3-11", "IGHV3-13", "IGHV3-20", "IGHV3-21",
    "IGHV3-23", "IGHV3-30", "IGHV3-33", "IGHV3-43", "IGHV3-48", "IGHV3-49",
    "IGHV3-53", "IGHV3-64", "IGHV3-66", "IGHV3-72", "IGHV3-73", "IGHV3-74",
    "IGHV4-4", "IGHV4-28", "IGHV4-30", "IGHV4-31", "IGHV4-34", "IGHV4-39",
    "IGHV4-59", "IGHV4-61",
    "IGHV5-51",
    "IGHV6-1",
    "IGHV7-4",
]

DH_GENES: list[str] = [
    "IGHD1-1", "IGHD1-7", "IGHD1-20", "IGHD1-26",
    "IGHD2-2", "IGHD2-15", "IGHD2-21",
    "IGHD3-3", "IGHD3-9", "IGHD3-10", "IGHD3-16", "IGHD3-22",
    "IGHD4-17", "IGHD4-23",
    "IGHD5-5", "IGHD5-12", "IGHD5-18", "IGHD5-24",
    "IGHD6-6", "IGHD6-13", "IGHD6-19", "IGHD6-25",
    "IGHD7-27",
]

JH_GENES: list[str] = [
    "IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6",
]

VL_GENES: list[str] = [
    "IGKV1-5", "IGKV1-9", "IGKV1-12", "IGKV1-16", "IGKV1-17", "IGKV1-27",
    "IGKV1-33", "IGKV1-39", "IGKV1-NL1",
    "IGKV2-24", "IGKV2-28", "IGKV2-29", "IGKV2-30", "IGKV2-40",
    "IGKV3-11", "IGKV3-15", "IGKV3-20",
    "IGKV4-1",
    "IGLV1-44", "IGLV1-47", "IGLV1-51",
    "IGLV2-8", "IGLV2-11", "IGLV2-14", "IGLV2-23",
    "IGLV3-1", "IGLV3-9", "IGLV3-10", "IGLV3-19", "IGLV3-21", "IGLV3-25",
]

JL_GENES: list[str] = [
    "IGKJ1", "IGKJ2", "IGKJ3", "IGKJ4", "IGKJ5",
    "IGLJ1", "IGLJ2", "IGLJ3",
]

ISOTYPES: list[str] = ["IgG1", "IgG2", "IgG3", "IgG4", "IgM", "IgA1", "IgA2", "IgE", "IgD"]
ISOTYPE_WEIGHTS: list[float] = [0.35, 0.15, 0.05, 0.08, 0.18, 0.09, 0.04, 0.03, 0.03]

EXPRESSION_SYSTEMS: list[str] = ["CHO", "HEK293", "Expi293"]
ASSAY_TYPES: list[str] = ["SPR", "ELISA", "neutralisation", "flow_binding"]
TARGET_ANTIGENS: list[str] = [
    "SARS-CoV-2-Spike", "CD20", "TNF-alpha", "IL-6", "PD-1", "HER2", "VEGF", "RSV-F",
]
STORAGE_CONDITIONS: list[str] = ["liquid_nitrogen", "DMSO", "-80C"]

COHORTS: list[str] = ["Healthy-Controls", "Autoimmune-Cohort", "Oncology-Cohort"]

PANEL_NAMES: list[str] = [
    "B-Cell-Immunophenotyping-v2",
    "Memory-B-Panel",
    "Plasmablast-Screen",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_date(start: date, end: date) -> date:
    """Return a random date between start and end (inclusive)."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def random_aa_seq(length: int) -> str:
    """Generate a random amino acid sequence of given length."""
    return "".join(random.choices(VALID_AA, k=length))


def make_content_hash(sequence: str) -> str:
    """Return first 32 hex chars of SHA-256 of sequence string."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:32]


def random_cdr3(min_len: int = 8, max_len: int = 20) -> str:
    """Generate a random CDR3 amino acid sequence."""
    length = random.randint(min_len, max_len)
    return random_aa_seq(length)


def build_full_vh(cdr1: str, cdr2: str, cdr3: str, vh_gene: str) -> str:
    """Build a plausible full VH amino acid string by combining framework and CDR regions.

    Uses a simplified framework scaffold with realistic lengths.
    """
    # Framework regions (fixed lengths, random sequence)
    fw1 = random_aa_seq(25)
    fw2 = random_aa_seq(17)
    fw3 = random_aa_seq(38)
    fw4 = random_aa_seq(11)
    return fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3 + fw4


def build_full_vl(cdr1: str, cdr2: str, cdr3: str) -> str:
    """Build a plausible full VL amino acid string."""
    fw1 = random_aa_seq(23)
    fw2 = random_aa_seq(15)
    fw3 = random_aa_seq(32)
    fw4 = random_aa_seq(10)
    return fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3 + fw4


def realistic_flow_percentages(disease_status: str) -> dict[str, float]:
    """Generate realistic flow cytometry B cell gate percentages.

    Healthy donors: typical B cell percentages.
    Autoimmune donors: elevated plasmablasts.
    Cancer/CLL donors: skewed B cell populations.

    Returns a dict of column_name -> value.
    """
    if disease_status == "healthy":
        b_cell_gate = round(random.uniform(5.0, 15.0), 2)
        naive_b = round(random.uniform(40.0, 60.0), 2)
        memory_b = round(random.uniform(25.0, 40.0), 2)
        plasmablast = round(random.uniform(2.0, 8.0), 2)
    elif disease_status in ("autoimmune/lupus", "autoimmune"):
        b_cell_gate = round(random.uniform(8.0, 20.0), 2)
        naive_b = round(random.uniform(30.0, 50.0), 2)
        memory_b = round(random.uniform(25.0, 45.0), 2)
        plasmablast = round(random.uniform(8.0, 20.0), 2)
    else:  # cancer/CLL
        b_cell_gate = round(random.uniform(15.0, 60.0), 2)
        naive_b = round(random.uniform(20.0, 40.0), 2)
        memory_b = round(random.uniform(15.0, 35.0), 2)
        plasmablast = round(random.uniform(1.0, 6.0), 2)

    # CD markers are subsets of the B cell gate, expressed as % of B cells
    cd19 = round(random.uniform(85.0, 98.0), 2)
    cd20 = round(random.uniform(70.0, 95.0), 2)
    cd38 = round(random.uniform(15.0, 40.0), 2)
    cd138 = round(random.uniform(0.5, 5.0), 2)

    # Ensure fractions roughly sum to 100% for subsets
    total_subset = naive_b + memory_b + plasmablast
    if total_subset > 100.0:
        factor = 95.0 / total_subset
        naive_b = round(naive_b * factor, 2)
        memory_b = round(memory_b * factor, 2)
        plasmablast = round(plasmablast * factor, 2)

    qc_pass = (
        b_cell_gate >= 3.0
        and cd19 >= 80.0
        and (naive_b + memory_b + plasmablast) <= 102.0
    )

    return {
        "b_cell_gate_pct": b_cell_gate,
        "cd19_pct": cd19,
        "cd20_pct": cd20,
        "cd38_pct": cd38,
        "cd138_pct": cd138,
        "naive_b_pct": naive_b,
        "memory_b_pct": memory_b,
        "plasmablast_pct": plasmablast,
        "qc_pass": qc_pass,
    }


# ---------------------------------------------------------------------------
# Seeder class
# ---------------------------------------------------------------------------

class LIMSSeeder:
    """Orchestrates synthetic data generation and insertion."""

    def __init__(self, db_url: str) -> None:
        """Initialise the seeder with a database connection URL.

        Args:
            db_url: SQLAlchemy-compatible database URL.
        """
        self.engine = create_engine(db_url, echo=False)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.session = self.SessionFactory()

        # Tracking containers filled during seeding
        self.users: list[User] = []
        self.instruments: list[Instrument] = []
        self.donors: list[Donor] = []
        self.samples: list[BCellSample] = []
        self.sequences: list[AntibodySequence] = []

    def close(self) -> None:
        """Close the database session."""
        self.session.close()

    # ------------------------------------------------------------------
    # Create tables
    # ------------------------------------------------------------------

    def ensure_tables(self) -> None:
        """Create all tables if they do not already exist."""
        Base.metadata.create_all(self.engine)
        print("[seed] Tables ensured.")

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def seed_users(self) -> None:
        """Insert 5 synthetic lab users (1 admin, 2 scientists, 2 analysts)."""
        user_specs: list[dict[str, Any]] = [
            {"username": "admin_alice", "email": "alice@lims.lab", "role": "admin"},
            {"username": "sci_bob", "email": "bob@lims.lab", "role": "scientist"},
            {"username": "sci_carol", "email": "carol@lims.lab", "role": "scientist"},
            {"username": "analyst_dave", "email": "dave@lims.lab", "role": "analyst"},
            {"username": "analyst_eve", "email": "eve@lims.lab", "role": "analyst"},
        ]
        for spec in user_specs:
            user = User(
                user_id=str(uuid.uuid4()),
                username=spec["username"],
                email=spec["email"],
                role=spec["role"],
                is_active=True,
                created_at=datetime(2023, 1, 1),
            )
            self.session.add(user)
            self.users.append(user)
        self.session.flush()
        print(f"[seed] Inserted {len(self.users)} users.")

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def seed_instruments(self) -> None:
        """Insert 4 lab instruments (2 flow cytometers, 1 plate reader, 1 BLI)."""
        instrument_specs: list[dict[str, Any]] = [
            {
                "name": "BD FACSAria Fusion 1",
                "instrument_type": "flow_cytometer",
                "manufacturer": "BD Biosciences",
                "model": "FACSAria Fusion",
                "serial_number": "SN-FACS-001",
                "calibration_date": date(2024, 1, 15),
            },
            {
                "name": "Sony SP6800 Spectral Analyser",
                "instrument_type": "flow_cytometer",
                "manufacturer": "Sony Biotechnology",
                "model": "SP6800",
                "serial_number": "SN-SONY-002",
                "calibration_date": date(2024, 2, 20),
            },
            {
                "name": "SpectraMax i3x Multi-Mode Reader",
                "instrument_type": "plate_reader",
                "manufacturer": "Molecular Devices",
                "model": "SpectraMax i3x",
                "serial_number": "SN-SMAX-003",
                "calibration_date": date(2024, 3, 10),
            },
            {
                "name": "Octet RED96e BLI System",
                "instrument_type": "biolayer_interferometry",
                "manufacturer": "Sartorius",
                "model": "Octet RED96",
                "serial_number": "SN-OCT-004",
                "calibration_date": date(2024, 1, 5),
            },
        ]
        for spec in instrument_specs:
            instrument = Instrument(
                instrument_id=str(uuid.uuid4()),
                name=spec["name"],
                instrument_type=spec["instrument_type"],
                manufacturer=spec["manufacturer"],
                model=spec["model"],
                serial_number=spec["serial_number"],
                calibration_date=spec["calibration_date"],
                is_active=True,
            )
            self.session.add(instrument)
            self.instruments.append(instrument)
        self.session.flush()
        print(f"[seed] Inserted {len(self.instruments)} instruments.")

    # ------------------------------------------------------------------
    # Donors
    # ------------------------------------------------------------------

    def seed_donors(self) -> None:
        """Insert 20 synthetic donors: 10 healthy, 5 autoimmune/lupus, 5 cancer/CLL."""
        donor_configs: list[tuple[str, str, str]] = (
            [("healthy", "Healthy-Controls", "healthy")] * 10
            + [("autoimmune/lupus", "Autoimmune-Cohort", "autoimmune")] * 5
            + [("cancer/CLL", "Oncology-Cohort", "cancer")] * 5
        )
        sexes = ["M", "F"]
        enroll_start = date(2021, 1, 1)
        enroll_end = date(2023, 12, 31)

        for i, (disease_status, cohort, _) in enumerate(donor_configs, start=1):
            donor = Donor(
                donor_id=str(uuid.uuid4()),
                donor_code=f"DONOR-{i:03d}",
                age=random.randint(25, 75),
                sex=random.choice(sexes),
                disease_status=disease_status,
                cohort=cohort,
                enrolled_at=random_date(enroll_start, enroll_end),
            )
            self.session.add(donor)
            self.donors.append(donor)
        self.session.flush()
        print(f"[seed] Inserted {len(self.donors)} donors.")

    # ------------------------------------------------------------------
    # B Cell Samples
    # ------------------------------------------------------------------

    def seed_samples(self) -> None:
        """Insert 3 B cell samples per donor (60 total)."""
        collector_users = [u for u in self.users if u.role in ("scientist",)]
        collect_start = date(2022, 1, 1)
        collect_end = date(2024, 6, 30)

        for donor in self.donors:
            for _ in range(3):
                sample = BCellSample(
                    sample_id=str(uuid.uuid4()),
                    donor_id=donor.donor_id,
                    collected_by=random.choice(collector_users).user_id if collector_users else None,
                    collection_date=random_date(collect_start, collect_end),
                    cell_count_1e6=round(random.uniform(1.0, 50.0), 2),
                    viability_pct=round(random.uniform(70.0, 99.0), 1),
                    storage_condition=random.choice(STORAGE_CONDITIONS),
                    notes=None if random.random() > 0.2 else "Sample collected under fasted conditions.",
                    created_at=datetime.utcnow(),
                )
                self.session.add(sample)
                self.samples.append(sample)
        self.session.flush()
        print(f"[seed] Inserted {len(self.samples)} B cell samples.")

    # ------------------------------------------------------------------
    # Flow Cytometry Runs
    # ------------------------------------------------------------------

    def seed_flow_runs(self) -> None:
        """Insert 2 flow cytometry runs per sample (120 total)."""
        flow_instruments = [i for i in self.instruments if i.instrument_type == "flow_cytometer"]
        operators = [u for u in self.users if u.role in ("scientist", "analyst")]

        count = 0
        for sample in self.samples:
            # Determine disease status from donor (look up through donor_id)
            donor = next(d for d in self.donors if d.donor_id == sample.donor_id)
            disease_status = donor.disease_status

            for _ in range(2):
                flow_pcts = realistic_flow_percentages(disease_status)
                run_date = random_date(
                    sample.collection_date or date(2022, 1, 1),
                    date(2024, 8, 31),
                )
                run = FlowCytometryRun(
                    run_id=str(uuid.uuid4()),
                    sample_id=sample.sample_id,
                    instrument_id=random.choice(flow_instruments).instrument_id,
                    operator_id=random.choice(operators).user_id if operators else None,
                    run_date=run_date,
                    panel_name=random.choice(PANEL_NAMES),
                    b_cell_gate_pct=flow_pcts["b_cell_gate_pct"],
                    cd19_pct=flow_pcts["cd19_pct"],
                    cd20_pct=flow_pcts["cd20_pct"],
                    cd38_pct=flow_pcts["cd38_pct"],
                    cd138_pct=flow_pcts["cd138_pct"],
                    naive_b_pct=flow_pcts["naive_b_pct"],
                    memory_b_pct=flow_pcts["memory_b_pct"],
                    plasmablast_pct=flow_pcts["plasmablast_pct"],
                    qc_pass=flow_pcts["qc_pass"],
                    created_at=datetime.utcnow(),
                )
                self.session.add(run)
                count += 1
        self.session.flush()
        print(f"[seed] Inserted {count} flow cytometry runs.")

    # ------------------------------------------------------------------
    # Antibody Sequences
    # ------------------------------------------------------------------

    def seed_sequences(self) -> None:
        """Insert ~8 antibody sequences per sample (~480 total).

        Sequences are assigned to clone groups (some expanded, most singleton).
        A content_hash based on full_vh_aa ensures deduplication.
        """
        seen_hashes: set[str] = set()
        count = 0

        for sample in self.samples:
            # Assign ~4 clones to this sample; some will be expanded
            n_clones = random.randint(3, 6)
            clone_ids_for_sample = [f"CLONE-{uuid.uuid4().hex[:8].upper()}" for _ in range(n_clones)]

            n_sequences = random.randint(6, 10)
            for _ in range(n_sequences):
                vh_gene = random.choice(VH_GENES)
                dh_gene = random.choice(DH_GENES)
                jh_gene = random.choice(JH_GENES)
                vl_gene = random.choice(VL_GENES)
                jl_gene = random.choice(JL_GENES)
                isotype = random.choices(ISOTYPES, weights=ISOTYPE_WEIGHTS)[0]

                cdr1 = random_aa_seq(random.randint(5, 12))
                cdr2 = random_aa_seq(random.randint(6, 10))
                cdr3 = random_cdr3(8, 20)
                full_vh = build_full_vh(cdr1, cdr2, cdr3, vh_gene)

                cdr1_l = random_aa_seq(random.randint(8, 14))
                cdr2_l = random_aa_seq(random.randint(3, 7))
                cdr3_l = random_cdr3(7, 11)
                full_vl = build_full_vl(cdr1_l, cdr2_l, cdr3_l)

                content_hash = make_content_hash(full_vh + full_vl)
                # Skip if duplicate (extremely rare with random sequences)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Expanded clone or singleton
                if random.random() < 0.7:
                    clone_id = random.choice(clone_ids_for_sample)
                    read_count = random.randint(2, 50)
                else:
                    clone_id = f"CLONE-{uuid.uuid4().hex[:8].upper()}"
                    read_count = 1

                seq = AntibodySequence(
                    seq_id=str(uuid.uuid4()),
                    sample_id=sample.sample_id,
                    chain_type="heavy",
                    vh_gene=vh_gene,
                    dh_gene=dh_gene,
                    jh_gene=jh_gene,
                    vl_gene=vl_gene,
                    jl_gene=jl_gene,
                    cdr1_aa=cdr1,
                    cdr2_aa=cdr2,
                    cdr3_aa=cdr3,
                    cdr3_length=len(cdr3),
                    full_vh_aa=full_vh,
                    full_vl_aa=full_vl,
                    isotype=isotype,
                    clone_id=clone_id,
                    read_count=read_count,
                    content_hash=content_hash,
                    created_at=datetime.utcnow(),
                )
                self.session.add(seq)
                self.sequences.append(seq)
                count += 1

        self.session.flush()
        print(f"[seed] Inserted {count} antibody sequences.")

    # ------------------------------------------------------------------
    # Expression Results
    # ------------------------------------------------------------------

    def seed_expression_results(self) -> None:
        """Insert 1-2 expression results for the top 100 sequences by read_count."""
        expresser_users = [u for u in self.users if u.role in ("scientist",)]
        top_seqs = sorted(self.sequences, key=lambda s: s.read_count, reverse=True)[:100]

        expr_start = date(2022, 6, 1)
        expr_end = date(2024, 9, 30)
        count = 0

        for seq in top_seqs:
            n_results = random.randint(1, 2)
            for run_idx in range(n_results):
                system = random.choice(EXPRESSION_SYSTEMS)
                # Realistic yield ranges by system
                if system == "Expi293":
                    yield_mg_l: float | None = round(random.uniform(50.0, 300.0), 1)
                elif system == "CHO":
                    yield_mg_l = round(random.uniform(20.0, 150.0), 1)
                else:  # HEK293
                    yield_mg_l = round(random.uniform(10.0, 80.0), 1)

                if random.random() < 0.05:
                    yield_mg_l = None  # failed expression

                result = ExpressionResult(
                    result_id=str(uuid.uuid4()),
                    seq_id=seq.seq_id,
                    construct_name=f"{seq.seq_id[:8]}-IgG1-{system}-run{run_idx + 1}",
                    expression_system=system,
                    yield_mg_l=yield_mg_l,
                    purity_pct=round(random.uniform(85.0, 99.5), 1) if yield_mg_l else None,
                    aggregation_pct=round(random.uniform(0.5, 8.0), 2) if yield_mg_l else None,
                    endotoxin_eu_ml=round(random.uniform(0.01, 2.0), 3) if yield_mg_l else None,
                    expression_date=random_date(expr_start, expr_end),
                    expressed_by=random.choice(expresser_users).user_id if expresser_users else None,
                    created_at=datetime.utcnow(),
                )
                self.session.add(result)
                count += 1

        self.session.flush()
        print(f"[seed] Inserted {count} expression results.")

    # ------------------------------------------------------------------
    # Assay Results
    # ------------------------------------------------------------------

    def seed_assay_results(self) -> None:
        """Insert 1-3 assay results for the top 200 sequences by read_count."""
        runner_users = [u for u in self.users if u.role in ("scientist", "analyst")]
        top_seqs = sorted(self.sequences, key=lambda s: s.read_count, reverse=True)[:200]

        assay_start = date(2022, 6, 1)
        assay_end = date(2024, 9, 30)
        count = 0

        for seq in top_seqs:
            n_assays = random.randint(1, 3)
            used_types: set[str] = set()

            for _ in range(n_assays):
                assay_type = random.choice(ASSAY_TYPES)
                if assay_type in used_types:
                    continue  # avoid duplicate assay type per sequence in this loop
                used_types.add(assay_type)

                target = random.choice(TARGET_ANTIGENS)

                # Realistic KD values: nM range; very tight binders in 0.1-10 nM
                binding_kd_nm: float | None = None
                ic50_nm: float | None = None
                neutralisation_pct: float | None = None
                pass_fail: str | None = None

                if assay_type == "SPR":
                    binding_kd_nm = round(10 ** random.uniform(-1.0, 2.5), 3)  # 0.1 – 300 nM
                    pass_fail = "pass" if binding_kd_nm < 50.0 else "fail"
                elif assay_type == "ELISA":
                    binding_kd_nm = round(10 ** random.uniform(0.0, 3.0), 2)
                    pass_fail = random.choices(["pass", "fail", "inconclusive"], weights=[0.6, 0.3, 0.1])[0]
                elif assay_type == "neutralisation":
                    ic50_nm = round(10 ** random.uniform(-0.5, 2.0), 3)  # 0.3 – 100 nM
                    neutralisation_pct = round(random.uniform(50.0, 100.0), 1)
                    pass_fail = "pass" if (ic50_nm < 20.0 and neutralisation_pct > 70.0) else "fail"
                elif assay_type == "flow_binding":
                    binding_kd_nm = round(10 ** random.uniform(0.5, 3.0), 2)
                    pass_fail = "pass" if binding_kd_nm < 200.0 else "fail"

                assay = AssayResult(
                    assay_id=str(uuid.uuid4()),
                    seq_id=seq.seq_id,
                    assay_type=assay_type,
                    target_antigen=target,
                    binding_kd_nm=binding_kd_nm,
                    ic50_nm=ic50_nm,
                    neutralisation_pct=neutralisation_pct,
                    pass_fail=pass_fail,
                    assay_date=random_date(assay_start, assay_end),
                    run_by=random.choice(runner_users).user_id if runner_users else None,
                    created_at=datetime.utcnow(),
                )
                self.session.add(assay)
                count += 1

        self.session.flush()
        print(f"[seed] Inserted {count} assay results.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full seeding pipeline and commit all data."""
        print("[seed] Starting LIMS seed data generation ...")
        self.ensure_tables()
        try:
            self.seed_users()
            self.seed_instruments()
            self.seed_donors()
            self.seed_samples()
            self.seed_flow_runs()
            self.seed_sequences()
            self.seed_expression_results()
            self.seed_assay_results()
            self.session.commit()
            print("[seed] All data committed successfully.")
        except Exception as exc:
            self.session.rollback()
            print(f"[seed] ERROR: {exc}")
            raise
        finally:
            self.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic seed data for the Antibody Discovery LIMS."
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="SQLAlchemy database URL (default: DATABASE_URL env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.db_url:
        print(
            "ERROR: No database URL provided. "
            "Set DATABASE_URL environment variable or pass --db-url.",
            file=sys.stderr,
        )
        sys.exit(1)

    seeder = LIMSSeeder(db_url=args.db_url)
    seeder.run()
