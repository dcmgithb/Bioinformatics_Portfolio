-- Assay Result Aggregation by Donor Cohort and Assay Type
-- Pass rates, median binding affinity, and neutralisation by cohort
WITH assay_with_donor AS (
    SELECT
        ar.assay_id,
        ar.assay_type,
        ar.target_antigen,
        ar.binding_kd_nm,
        ar.ic50_nm,
        ar.neutralisation_pct,
        ar.pass_fail,
        ar.assay_date,
        d.cohort,
        d.disease_status,
        d.donor_code,
        ab.vh_gene,
        ab.isotype,
        REGEXP_REPLACE(ab.vh_gene, '-.*$', '') AS vh_family
    FROM assay_results ar
    JOIN antibody_sequences ab ON ar.seq_id = ab.seq_id
    JOIN b_cell_samples s      ON ab.sample_id = s.sample_id
    JOIN donors d              ON s.donor_id = d.donor_id
),
cohort_summary AS (
    SELECT
        cohort,
        disease_status,
        assay_type,
        target_antigen,
        COUNT(*)                                       AS n_assays,
        SUM(CASE WHEN pass_fail = 'pass' THEN 1 ELSE 0 END) AS n_pass,
        ROUND(
            SUM(CASE WHEN pass_fail = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            1
        )                                              AS pass_rate_pct,
        ROUND(
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY binding_kd_nm)::numeric, 3
        )                                              AS median_kd_nm,
        ROUND(AVG(binding_kd_nm)::numeric, 3)          AS mean_kd_nm,
        ROUND(
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ic50_nm)::numeric, 2
        )                                              AS median_ic50_nm,
        ROUND(AVG(neutralisation_pct)::numeric, 1)     AS mean_neutralisation_pct,
        COUNT(DISTINCT donor_code)                     AS n_donors
    FROM assay_with_donor
    GROUP BY cohort, disease_status, assay_type, target_antigen
)
SELECT
    cohort,
    disease_status,
    assay_type,
    target_antigen,
    n_assays,
    n_pass,
    pass_rate_pct,
    median_kd_nm,
    mean_kd_nm,
    median_ic50_nm,
    mean_neutralisation_pct,
    n_donors
FROM cohort_summary
ORDER BY cohort, assay_type, pass_rate_pct DESC;
