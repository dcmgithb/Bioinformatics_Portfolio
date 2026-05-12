-- Expression Yield by Construct and System
-- Ranks constructs by mean yield; includes statistical summary per expression system
WITH yield_stats AS (
    SELECT
        er.construct_name,
        er.expression_system,
        COUNT(*)                             AS n_runs,
        ROUND(AVG(er.yield_mg_l)::numeric, 2)  AS mean_yield_mg_l,
        ROUND(STDDEV(er.yield_mg_l)::numeric, 2) AS stddev_yield,
        ROUND(MIN(er.yield_mg_l)::numeric, 2)  AS min_yield,
        ROUND(MAX(er.yield_mg_l)::numeric, 2)  AS max_yield,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY er.yield_mg_l)::numeric, 2) AS median_yield,
        ROUND(AVG(er.purity_pct)::numeric, 1) AS mean_purity_pct,
        ROUND(AVG(er.aggregation_pct)::numeric, 1) AS mean_aggregation_pct
    FROM expression_results er
    WHERE er.yield_mg_l IS NOT NULL
    GROUP BY er.construct_name, er.expression_system
    HAVING COUNT(*) >= 2
),
ranked AS (
    SELECT
        *,
        RANK() OVER (PARTITION BY expression_system ORDER BY mean_yield_mg_l DESC) AS rank_within_system,
        RANK() OVER (ORDER BY mean_yield_mg_l DESC) AS rank_overall
    FROM yield_stats
)
SELECT *
FROM ranked
ORDER BY rank_overall
LIMIT 30;
