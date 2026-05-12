-- VDJ Gene Usage Statistics
-- Counts antibody sequences per VH gene, computes frequency %, and groups by IGHV family
-- Usage: Run against lims database with antibody_sequences table populated
WITH vh_counts AS (
    SELECT
        vh_gene,
        COUNT(*) AS sequence_count,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS frequency_pct
    FROM antibody_sequences
    WHERE vh_gene IS NOT NULL
    GROUP BY vh_gene
),
family_counts AS (
    SELECT
        REGEXP_REPLACE(vh_gene, '-.*$', '') AS vh_family,
        SUM(sequence_count) AS family_count,
        SUM(frequency_pct) AS family_pct
    FROM vh_counts
    GROUP BY vh_family
)
SELECT
    vc.vh_gene,
    vc.sequence_count,
    ROUND(vc.frequency_pct::numeric, 2) AS frequency_pct,
    fc.vh_family,
    fc.family_count,
    ROUND(fc.family_pct::numeric, 2) AS family_pct,
    RANK() OVER (ORDER BY vc.sequence_count DESC) AS rank
FROM vh_counts vc
JOIN family_counts fc ON REGEXP_REPLACE(vc.vh_gene, '-.*$', '') = fc.vh_family
ORDER BY vc.sequence_count DESC
LIMIT 25;
