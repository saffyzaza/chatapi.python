-- ============================================================
-- Migration 030: Populate mart_province_road from base tables
-- Run once after initial data load.
-- ============================================================

-- Classify road type from dim_road_segment.road_type text
-- สายหลัก = ทางหลวงแผ่นดิน / national highway
-- สายรอง  = ทางหลวงชนบท / rural highway
-- ไม่ระบุ = anything else / NULL

TRUNCATE mart_province_road RESTART IDENTITY;

INSERT INTO mart_province_road (
    year_no, geography_id, province_name, district_name,
    road_name, road_code, road_type_label,
    accident_count, injured_count, serious_injured, death_count,
    hotspot_score, dominant_cause, dominant_vehicle
)
SELECT
    e.csv_year                                                        AS year_no,
    g.geography_id,
    g.province_name,
    g.district_name,
    COALESCE(rs.road_name, 'ไม่ระบุ')                                AS road_name,
    COALESCE(rs.road_code, '')                                        AS road_code,

    -- Road type classification
    CASE
        WHEN COALESCE(rs.road_type,'') ILIKE ANY(ARRAY[
                '%ทางหลวงแผ่นดิน%','%สายหลัก%','%national%','% ทล.%'])
          OR COALESCE(e.accident_location,'') ILIKE ANY(ARRAY[
                '%ทางหลวงแผ่นดิน%','%สายหลัก%','% ทล.%'])
            THEN 'สายหลัก'
        WHEN COALESCE(rs.road_type,'') ILIKE ANY(ARRAY[
                '%ทางหลวงชนบท%','%สายรอง%','%rural%','% ทช.%'])
          OR COALESCE(e.accident_location,'') ILIKE ANY(ARRAY[
                '%ทางหลวงชนบท%','%สายรอง%','% ทช.%'])
            THEN 'สายรอง'
        ELSE 'ไม่ระบุ'
    END                                                               AS road_type_label,

    COUNT(*)                                                          AS accident_count,
    COALESCE(SUM(e.injured_count),    0)                             AS injured_count,
    COALESCE(SUM(e.serious_injured),  0)                             AS serious_injured,
    COALESCE(SUM(e.death_count),      0)                             AS death_count,

    -- Hotspot score: deaths×3 + serious×1.5 + accidents×1
    (COALESCE(SUM(e.death_count),0) * 3.0
     + COALESCE(SUM(e.serious_injured),0) * 1.5
     + COUNT(*) * 1.0)                                               AS hotspot_score,

    MODE() WITHIN GROUP (ORDER BY e.cause_presumed)                  AS dominant_cause,
    MODE() WITHIN GROUP (ORDER BY e.vehicle_type)                    AS dominant_vehicle

FROM fact_accident_event   e
JOIN  dim_geography         g  ON e.geography_id    = g.geography_id
LEFT JOIN dim_road_segment rs  ON e.road_segment_id = rs.road_segment_id
WHERE e.csv_year IS NOT NULL
  AND g.province_name IS NOT NULL
GROUP BY
    e.csv_year, g.geography_id, g.province_name, g.district_name,
    road_name, road_code, road_type_label
ON CONFLICT DO NOTHING;

-- Verify
SELECT road_type_label, COUNT(*) AS rows, SUM(death_count) AS total_deaths
FROM mart_province_road
GROUP BY road_type_label
ORDER BY road_type_label;
