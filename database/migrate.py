"""Database migration runner — creates all tables from schema.sql."""
import sys
import os
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import get_settings


def run_migration():
    s = get_settings()
    schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")

    print(f"Connecting to {s.DB_HOST}:{s.DB_PORT}/{s.DB_NAME} as {s.DB_USER}...")
    conn = psycopg2.connect(
        host=s.DB_HOST, port=s.DB_PORT,
        database=s.DB_NAME, user=s.DB_USER, password=s.DB_PASSWORD,
    )
    conn.autocommit = True

    with open(schema_file, encoding="utf-8") as f:
        sql = f.read()

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()

    print("Migration completed successfully.")
    print("\nTables created:")
    print("  - dim_geography")
    print("  - dim_time")
    print("  - dim_source")
    print("  - dim_road_segment")
    print("  - fact_accident_event")
    print("  - fact_accident_person  (empty — no person-level data)")
    print("  - mart_accident_summary")
    print("  - mart_accident_hotspot")
    print("  - mart_province_year")
    print("  - mart_province_road")
    print("\nNext step: run import_csv.py to load accident CSV data.")


if __name__ == "__main__":
    run_migration()
