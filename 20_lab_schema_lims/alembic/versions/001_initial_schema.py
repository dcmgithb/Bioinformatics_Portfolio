"""Initial database schema for the Antibody Discovery LIMS.

Creates all 9 tables:
    users, instruments, donors, b_cell_samples, flow_cytometry_runs,
    antibody_sequences, expression_results, assay_results, audit_log

Revision ID: a1b2c3d4e5f6
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Revision identifiers (used by Alembic)
# ---------------------------------------------------------------------------

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


# ---------------------------------------------------------------------------
# Upgrade — create all tables and indexes
# ---------------------------------------------------------------------------

def upgrade() -> None:
    """Create all LIMS tables with columns, foreign keys, and indexes."""

    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("email", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    # ------------------------------------------------------------------
    # instruments
    # ------------------------------------------------------------------
    op.create_table(
        "instruments",
        sa.Column("instrument_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("instrument_type", sa.String(64), nullable=False),
        sa.Column("manufacturer", sa.String(128), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("serial_number", sa.String(64), nullable=False),
        sa.Column("calibration_date", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.UniqueConstraint("serial_number", name="uq_instruments_serial_number"),
    )

    # ------------------------------------------------------------------
    # donors
    # ------------------------------------------------------------------
    op.create_table(
        "donors",
        sa.Column("donor_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("donor_code", sa.String(32), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("sex", sa.String(1), nullable=False),
        sa.Column("disease_status", sa.String(64), nullable=False),
        sa.Column("cohort", sa.String(128), nullable=False),
        sa.Column("enrolled_at", sa.Date(), nullable=True),
        sa.UniqueConstraint("donor_code", name="uq_donors_donor_code"),
    )

    # ------------------------------------------------------------------
    # b_cell_samples
    # ------------------------------------------------------------------
    op.create_table(
        "b_cell_samples",
        sa.Column("sample_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("donor_id", sa.String(36), nullable=False),
        sa.Column("collected_by", sa.String(36), nullable=True),
        sa.Column("collection_date", sa.Date(), nullable=True),
        sa.Column("cell_count_1e6", sa.Float(), nullable=True),
        sa.Column("viability_pct", sa.Float(), nullable=True),
        sa.Column("storage_condition", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["donor_id"], ["donors.donor_id"], name="fk_b_cell_samples_donor_id"
        ),
        sa.ForeignKeyConstraint(
            ["collected_by"], ["users.user_id"], name="fk_b_cell_samples_collected_by"
        ),
    )
    op.create_index("idx_bcell_samples_donor_id", "b_cell_samples", ["donor_id"])
    op.create_index(
        "idx_bcell_samples_collection_date", "b_cell_samples", ["collection_date"]
    )

    # ------------------------------------------------------------------
    # flow_cytometry_runs
    # ------------------------------------------------------------------
    op.create_table(
        "flow_cytometry_runs",
        sa.Column("run_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("sample_id", sa.String(36), nullable=False),
        sa.Column("instrument_id", sa.String(36), nullable=False),
        sa.Column("operator_id", sa.String(36), nullable=True),
        sa.Column("run_date", sa.Date(), nullable=True),
        sa.Column("panel_name", sa.String(128), nullable=False),
        sa.Column("b_cell_gate_pct", sa.Float(), nullable=True),
        sa.Column("cd19_pct", sa.Float(), nullable=True),
        sa.Column("cd20_pct", sa.Float(), nullable=True),
        sa.Column("cd38_pct", sa.Float(), nullable=True),
        sa.Column("cd138_pct", sa.Float(), nullable=True),
        sa.Column("naive_b_pct", sa.Float(), nullable=True),
        sa.Column("memory_b_pct", sa.Float(), nullable=True),
        sa.Column("plasmablast_pct", sa.Float(), nullable=True),
        sa.Column("qc_pass", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["b_cell_samples.sample_id"],
            name="fk_flow_runs_sample_id",
        ),
        sa.ForeignKeyConstraint(
            ["instrument_id"],
            ["instruments.instrument_id"],
            name="fk_flow_runs_instrument_id",
        ),
        sa.ForeignKeyConstraint(
            ["operator_id"], ["users.user_id"], name="fk_flow_runs_operator_id"
        ),
    )
    op.create_index("idx_flow_runs_sample_id", "flow_cytometry_runs", ["sample_id"])
    op.create_index("idx_flow_runs_run_date", "flow_cytometry_runs", ["run_date"])

    # ------------------------------------------------------------------
    # antibody_sequences
    # ------------------------------------------------------------------
    op.create_table(
        "antibody_sequences",
        sa.Column("seq_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("sample_id", sa.String(36), nullable=False),
        sa.Column("chain_type", sa.String(16), nullable=False),
        sa.Column("vh_gene", sa.String(32), nullable=True),
        sa.Column("dh_gene", sa.String(32), nullable=True),
        sa.Column("jh_gene", sa.String(16), nullable=True),
        sa.Column("vl_gene", sa.String(32), nullable=True),
        sa.Column("jl_gene", sa.String(16), nullable=True),
        sa.Column("cdr1_aa", sa.String(64), nullable=True),
        sa.Column("cdr2_aa", sa.String(64), nullable=True),
        sa.Column("cdr3_aa", sa.String(64), nullable=True),
        sa.Column("cdr3_length", sa.Integer(), nullable=True),
        sa.Column("full_vh_aa", sa.Text(), nullable=True),
        sa.Column("full_vl_aa", sa.Text(), nullable=True),
        sa.Column("isotype", sa.String(16), nullable=True),
        sa.Column("clone_id", sa.String(64), nullable=True),
        sa.Column("read_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["b_cell_samples.sample_id"],
            name="fk_ab_seqs_sample_id",
        ),
        sa.UniqueConstraint("content_hash", name="uq_ab_seqs_content_hash"),
    )
    op.create_index("idx_ab_seqs_sample_id", "antibody_sequences", ["sample_id"])
    op.create_index("idx_ab_seqs_clone_id", "antibody_sequences", ["clone_id"])
    op.create_index("idx_ab_seqs_vh_gene", "antibody_sequences", ["vh_gene"])
    op.create_index("idx_ab_seqs_cdr3_length", "antibody_sequences", ["cdr3_length"])

    # ------------------------------------------------------------------
    # expression_results
    # ------------------------------------------------------------------
    op.create_table(
        "expression_results",
        sa.Column("result_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("seq_id", sa.String(36), nullable=False),
        sa.Column("construct_name", sa.String(128), nullable=False),
        sa.Column("expression_system", sa.String(32), nullable=False),
        sa.Column("yield_mg_l", sa.Float(), nullable=True),
        sa.Column("purity_pct", sa.Float(), nullable=True),
        sa.Column("aggregation_pct", sa.Float(), nullable=True),
        sa.Column("endotoxin_eu_ml", sa.Float(), nullable=True),
        sa.Column("expression_date", sa.Date(), nullable=True),
        sa.Column("expressed_by", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["seq_id"],
            ["antibody_sequences.seq_id"],
            name="fk_expr_results_seq_id",
        ),
        sa.ForeignKeyConstraint(
            ["expressed_by"], ["users.user_id"], name="fk_expr_results_expressed_by"
        ),
    )
    op.create_index("idx_expr_results_seq_id", "expression_results", ["seq_id"])
    op.create_index(
        "idx_expr_results_construct_name", "expression_results", ["construct_name"]
    )

    # ------------------------------------------------------------------
    # assay_results
    # ------------------------------------------------------------------
    op.create_table(
        "assay_results",
        sa.Column("assay_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("seq_id", sa.String(36), nullable=False),
        sa.Column("assay_type", sa.String(32), nullable=False),
        sa.Column("target_antigen", sa.String(128), nullable=False),
        sa.Column("binding_kd_nm", sa.Float(), nullable=True),
        sa.Column("ic50_nm", sa.Float(), nullable=True),
        sa.Column("neutralisation_pct", sa.Float(), nullable=True),
        sa.Column("pass_fail", sa.String(16), nullable=True),
        sa.Column("assay_date", sa.Date(), nullable=True),
        sa.Column("run_by", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["seq_id"],
            ["antibody_sequences.seq_id"],
            name="fk_assay_results_seq_id",
        ),
        sa.ForeignKeyConstraint(
            ["run_by"], ["users.user_id"], name="fk_assay_results_run_by"
        ),
    )
    op.create_index("idx_assay_results_seq_id", "assay_results", ["seq_id"])
    op.create_index("idx_assay_results_assay_type", "assay_results", ["assay_type"])
    op.create_index("idx_assay_results_pass_fail", "assay_results", ["pass_fail"])

    # ------------------------------------------------------------------
    # audit_log
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("log_id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("table_name", sa.String(64), nullable=False),
        sa.Column("record_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("changed_by", sa.String(36), nullable=True),
        sa.Column(
            "changed_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("old_values", sa.JSON(), nullable=True),
        sa.Column("new_values", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["changed_by"], ["users.user_id"], name="fk_audit_log_changed_by"
        ),
    )
    op.create_index(
        "idx_audit_log_table_record", "audit_log", ["table_name", "record_id"]
    )
    op.create_index("idx_audit_log_changed_at", "audit_log", ["changed_at"])


# ---------------------------------------------------------------------------
# Downgrade — drop all tables in reverse dependency order
# ---------------------------------------------------------------------------

def downgrade() -> None:
    """Drop all LIMS tables and their indexes."""

    # Drop indexes then tables (indexes are dropped automatically with tables in
    # most databases, but we list them explicitly for clarity / SQLite compat).
    op.drop_index("idx_audit_log_changed_at", table_name="audit_log")
    op.drop_index("idx_audit_log_table_record", table_name="audit_log")
    op.drop_table("audit_log")

    op.drop_index("idx_assay_results_pass_fail", table_name="assay_results")
    op.drop_index("idx_assay_results_assay_type", table_name="assay_results")
    op.drop_index("idx_assay_results_seq_id", table_name="assay_results")
    op.drop_table("assay_results")

    op.drop_index("idx_expr_results_construct_name", table_name="expression_results")
    op.drop_index("idx_expr_results_seq_id", table_name="expression_results")
    op.drop_table("expression_results")

    op.drop_index("idx_ab_seqs_cdr3_length", table_name="antibody_sequences")
    op.drop_index("idx_ab_seqs_vh_gene", table_name="antibody_sequences")
    op.drop_index("idx_ab_seqs_clone_id", table_name="antibody_sequences")
    op.drop_index("idx_ab_seqs_sample_id", table_name="antibody_sequences")
    op.drop_table("antibody_sequences")

    op.drop_index("idx_flow_runs_run_date", table_name="flow_cytometry_runs")
    op.drop_index("idx_flow_runs_sample_id", table_name="flow_cytometry_runs")
    op.drop_table("flow_cytometry_runs")

    op.drop_index("idx_bcell_samples_collection_date", table_name="b_cell_samples")
    op.drop_index("idx_bcell_samples_donor_id", table_name="b_cell_samples")
    op.drop_table("b_cell_samples")

    op.drop_table("donors")
    op.drop_table("instruments")
    op.drop_table("users")
