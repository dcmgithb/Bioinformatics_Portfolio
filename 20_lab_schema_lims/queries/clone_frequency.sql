-- Clone Frequency Analysis
-- Identifies expanded B cell clones by clone_id and computes diversity metrics
WITH clone_stats AS (
    SELECT
        clone_id,
        COUNT(DISTINCT seq_id)  AS unique_sequences,
        SUM(read_count)         AS total_reads,
        SUM(SUM(read_count)) OVER () AS grand_total_reads,
        MIN(cdr3_length)        AS min_cdr3_length,
        MAX(cdr3_length)        AS max_cdr3_length,
        MODE() WITHIN GROUP (ORDER BY isotype) AS dominant_isotype,
        MODE() WITHIN GROUP (ORDER BY vh_gene) AS dominant_vh_gene
    FROM antibody_sequences
    WHERE clone_id IS NOT NULL
    GROUP BY clone_id
),
clone_freq AS (
    SELECT
        clone_id,
        unique_sequences,
        total_reads,
        ROUND((total_reads * 100.0 / grand_total_reads)::numeric, 4) AS clonal_frequency_pct,
        min_cdr3_length,
        max_cdr3_length,
        dominant_isotype,
        dominant_vh_gene,
        -- Per-clone contribution to Shannon entropy: -p*ln(p)
        -(total_reads::float / grand_total_reads) * LN(total_reads::float / grand_total_reads) AS shannon_contrib
    FROM clone_stats
)
SELECT
    clone_id,
    unique_sequences,
    total_reads,
    clonal_frequency_pct,
    dominant_isotype,
    dominant_vh_gene,
    min_cdr3_length,
    max_cdr3_length,
    RANK() OVER (ORDER BY total_reads DESC) AS clone_rank,
    ROUND(SUM(shannon_contrib) OVER ()::numeric, 4) AS shannon_diversity_index
FROM clone_freq
ORDER BY total_reads DESC
LIMIT 50;
