-- Accident Agent — Complete Database Schema
-- Run this file once to initialize the database.
-- Order matters: shared dims → accident domain → mart tables → indexes

-- ────────────────────────────────────────────────────────────────────────────
-- SHARED DIMENSIONS
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_geography (
    geography_id    BIGSERIAL PRIMARY KEY,
    country_code    VARCHAR(10)  DEFAULT 'TH',
    province_code   VARCHAR(20),
    province_name   VARCHAR(255),
    district_code   VARCHAR(20),
    district_name   VARCHAR(255),
    subdistrict_code  VARCHAR(20),
    subdistrict_name  VARCHAR(255),
    latitude        DECIMAL(10,6),
    longitude       DECIMAL(10,6)
);

CREATE INDEX IF NOT EXISTS idx_geo_province ON dim_geography(province_name);
CREATE INDEX IF NOT EXISTS idx_geo_district ON dim_geography(district_name);

CREATE TABLE IF NOT EXISTS dim_time (
    time_id      BIGSERIAL PRIMARY KEY,
    full_date    DATE UNIQUE,
    day_of_week  VARCHAR(20),
    week_no      INT,
    month_no     INT,
    month_name   VARCHAR(20),
    quarter_no   INT,
    year_no      INT,
    hour_no      INT
);

CREATE INDEX IF NOT EXISTS idx_time_date ON dim_time(full_date);
CREATE INDEX IF NOT EXISTS idx_time_year_month ON dim_time(year_no, month_no);

CREATE TABLE IF NOT EXISTS dim_source (
    source_id        BIGSERIAL PRIMARY KEY,
    source_name      VARCHAR(255),
    source_type      VARCHAR(100),
    owner_org        VARCHAR(255),
    update_frequency VARCHAR(50),
    quality_level    VARCHAR(50)
);

-- Populate time dimension 2020-2030
INSERT INTO dim_time (full_date, day_of_week, week_no, month_no, month_name, quarter_no, year_no, hour_no)
SELECT d::DATE,
       TO_CHAR(d, 'Day'),
       EXTRACT(WEEK FROM d)::INT,
       EXTRACT(MONTH FROM d)::INT,
       TO_CHAR(d, 'Month'),
       EXTRACT(QUARTER FROM d)::INT,
       EXTRACT(YEAR FROM d)::INT,
       0
FROM generate_series('2020-01-01'::DATE, '2030-12-31'::DATE, '1 day'::INTERVAL) AS d
ON CONFLICT (full_date) DO NOTHING;


-- ────────────────────────────────────────────────────────────────────────────
-- ACCIDENT DOMAIN
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_road_segment (
    road_segment_id BIGSERIAL PRIMARY KEY,
    road_name       VARCHAR(255),
    road_type       VARCHAR(100),
    road_code       VARCHAR(50),
    geography_id    BIGINT REFERENCES dim_geography(geography_id),
    km_marker       DECIMAL(10,3),
    cause_type      VARCHAR(255),
    lane_count      INT,
    curvature_type  VARCHAR(100),
    slope_type      VARCHAR(100),
    speed_limit     INT,
    surface_type    VARCHAR(100),
    risk_flag       BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_road_seg_geo  ON dim_road_segment(geography_id);
CREATE INDEX IF NOT EXISTS idx_road_seg_code ON dim_road_segment(road_code);

CREATE TABLE IF NOT EXISTS fact_accident_event (
    accident_id      BIGSERIAL PRIMARY KEY,
    event_datetime   TIMESTAMP,
    geography_id     BIGINT REFERENCES dim_geography(geography_id),
    road_segment_id  BIGINT REFERENCES dim_road_segment(road_segment_id),
    weather_condition   VARCHAR(100),
    accident_type       VARCHAR(100),
    accident_location   VARCHAR(255),
    cause_presumed      VARCHAR(255),
    severity_level      VARCHAR(50),
    vehicle_type        VARCHAR(100),
    injured_count       INT DEFAULT 0,
    serious_injured     INT DEFAULT 0,
    death_count         INT DEFAULT 0,
    csv_year            INT,
    source_id           BIGINT REFERENCES dim_source(source_id)
);

CREATE INDEX IF NOT EXISTS idx_accident_datetime  ON fact_accident_event(event_datetime);
CREATE INDEX IF NOT EXISTS idx_accident_geo        ON fact_accident_event(geography_id);
CREATE INDEX IF NOT EXISTS idx_accident_severity   ON fact_accident_event(severity_level);
CREATE INDEX IF NOT EXISTS idx_accident_csvyear    ON fact_accident_event(csv_year);
CREATE INDEX IF NOT EXISTS idx_accident_location   ON fact_accident_event(accident_location);
CREATE INDEX IF NOT EXISTS idx_accident_cause      ON fact_accident_event(cause_presumed);

CREATE TABLE IF NOT EXISTS fact_accident_person (
    person_event_id BIGSERIAL PRIMARY KEY,
    accident_id     BIGINT REFERENCES fact_accident_event(accident_id),
    age             INT,
    sex             VARCHAR(20),
    role_in_event   VARCHAR(100),
    injury_level    VARCHAR(100),
    helmet_used     BOOLEAN,
    seatbelt_used   BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_acc_person_accident ON fact_accident_person(accident_id);


-- ────────────────────────────────────────────────────────────────────────────
-- ANALYTIC MART TABLES
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mart_accident_summary (
    id                  BIGSERIAL PRIMARY KEY,
    year_no             INT NOT NULL,
    month_no            INT NOT NULL,
    geography_id        BIGINT REFERENCES dim_geography(geography_id),
    province_name       VARCHAR(255),
    accident_count      INT DEFAULT 0,
    injured_count       INT DEFAULT 0,
    death_count         INT DEFAULT 0,
    high_risk_timeband  VARCHAR(100),
    dominant_road_cond  VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_mart_acc_ym   ON mart_accident_summary(year_no, month_no);
CREATE INDEX IF NOT EXISTS idx_mart_acc_geo  ON mart_accident_summary(geography_id);
CREATE INDEX IF NOT EXISTS idx_mart_summary_prov ON mart_accident_summary(province_name);

CREATE TABLE IF NOT EXISTS mart_accident_hotspot (
    hotspot_id       BIGSERIAL PRIMARY KEY,
    geography_id     BIGINT REFERENCES dim_geography(geography_id),
    road_segment_id  BIGINT REFERENCES dim_road_segment(road_segment_id),
    accident_count   INT DEFAULT 0,
    injured_count    INT DEFAULT 0,
    death_count      INT DEFAULT 0,
    hotspot_score    DECIMAL(10,2) DEFAULT 0,
    dominant_timeband VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_hotspot_score ON mart_accident_hotspot(hotspot_score DESC);

-- Yearly summary by province
CREATE TABLE IF NOT EXISTS mart_province_year (
    id              BIGSERIAL PRIMARY KEY,
    year_no         INT NOT NULL,
    geography_id    BIGINT REFERENCES dim_geography(geography_id),
    province_name   VARCHAR(255),
    accident_count  INT DEFAULT 0,
    injured_count   INT DEFAULT 0,
    death_count     INT DEFAULT 0,
    serious_injured INT DEFAULT 0,
    road_count      INT DEFAULT 0,
    top_vehicle     VARCHAR(100),
    top_cause       VARCHAR(255),
    top_timeband    VARCHAR(100),
    top_weather     VARCHAR(100)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_province_year_uniq ON mart_province_year(year_no, geography_id);
CREATE INDEX IF NOT EXISTS idx_province_year_prov ON mart_province_year(province_name);
CREATE INDEX IF NOT EXISTS idx_province_year_yr   ON mart_province_year(year_no);

-- Road accident breakdown per province per year
CREATE TABLE IF NOT EXISTS mart_province_road (
    id               BIGSERIAL PRIMARY KEY,
    year_no          INT NOT NULL,
    geography_id     BIGINT REFERENCES dim_geography(geography_id),
    province_name    VARCHAR(255),
    district_name    VARCHAR(255),
    road_segment_id  BIGINT REFERENCES dim_road_segment(road_segment_id),
    road_name        VARCHAR(255),
    road_code        VARCHAR(50),
    road_type_label  VARCHAR(50),
    accident_count   INT DEFAULT 0,
    injured_count    INT DEFAULT 0,
    death_count      INT DEFAULT 0,
    serious_injured  INT DEFAULT 0,
    hotspot_score    DECIMAL(12,2) DEFAULT 0,
    dominant_cause   VARCHAR(255),
    dominant_vehicle VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_prov_road_geo_yr   ON mart_province_road(geography_id, year_no);
CREATE INDEX IF NOT EXISTS idx_prov_road_prov     ON mart_province_road(province_name);
CREATE INDEX IF NOT EXISTS idx_prov_road_score    ON mart_province_road(hotspot_score DESC);
CREATE INDEX IF NOT EXISTS idx_prov_road_district ON mart_province_road(district_name);


-- ────────────────────────────────────────────────────────────────────────────
-- VIEWS
-- ────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_province_year_summary AS
SELECT py.year_no, py.province_name,
       py.accident_count, py.injured_count, py.serious_injured,
       py.death_count, py.road_count, py.top_vehicle, py.top_cause,
       py.top_timeband, py.top_weather, g.latitude, g.longitude
FROM mart_province_year py
LEFT JOIN dim_geography g ON py.geography_id = g.geography_id;

CREATE OR REPLACE VIEW v_province_road_year AS
SELECT pr.year_no, pr.province_name, pr.district_name,
       pr.road_name, pr.road_code, pr.road_type_label,
       pr.accident_count, pr.injured_count, pr.serious_injured,
       pr.death_count, pr.hotspot_score, pr.dominant_cause, pr.dominant_vehicle
FROM mart_province_road pr;


CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  name             VARCHAR(255) NOT NULL,
  email            VARCHAR(255) UNIQUE NOT NULL,
  password_hash    VARCHAR(255) NOT NULL,
  role             VARCHAR(50)  NOT NULL DEFAULT 'user',       -- 'user' | 'admin' | 'adminsuper'
  status           VARCHAR(50)  NOT NULL DEFAULT 'pending',    -- 'pending' | 'approved' | 'rejected'
  approved_by      UUID         REFERENCES accounts(id),
  approved_at      TIMESTAMPTZ,

  -- Profile fields
  prefix           VARCHAR(50),
  organization     VARCHAR(255),
  position         VARCHAR(255),
  department       VARCHAR(255),
  phone            VARCHAR(50),
  province         VARCHAR(100),
  health_zone      VARCHAR(100),
  parent_organization VARCHAR(255),
  org_code         VARCHAR(100),
  address          TEXT,
  website          VARCHAR(255),

  -- Password reset
  reset_token          VARCHAR(255),
  reset_token_expires  TIMESTAMPTZ,

  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS accounts_email_idx        ON accounts(email);
CREATE INDEX IF NOT EXISTS accounts_status_idx       ON accounts(status);
CREATE INDEX IF NOT EXISTS accounts_reset_token_idx  ON accounts(reset_token);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS accounts_updated_at ON accounts;
CREATE TRIGGER accounts_updated_at
  BEFORE UPDATE ON accounts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Chat session history
CREATE TABLE IF NOT EXISTS chat_sessions (
  id               UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id       TEXT         NOT NULL UNIQUE,
  user_id          UUID         REFERENCES accounts(id) ON DELETE SET NULL,
  status           VARCHAR(20)  NOT NULL DEFAULT 'idle',
  last_user_prompt TEXT,
  messages_json    JSONB        NOT NULL DEFAULT '[]'::jsonb,
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_sessions_user_id_idx ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS chat_sessions_updated_at_idx ON chat_sessions(updated_at DESC);

DROP TRIGGER IF EXISTS chat_sessions_updated_at ON chat_sessions;
CREATE TRIGGER chat_sessions_updated_at
  BEFORE UPDATE ON chat_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Seed first admin account (change password after first login)
-- INSERT INTO accounts (name, email, password_hash, role, status)
-- VALUES ('Admin', 'admin@example.com', '$2a$12$...', 'admin', 'approved');

-- Journal Reports — saved HTML reports from ThaiJo pipeline
CREATE TABLE IF NOT EXISTS journal_reports (
  id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID         NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  title         TEXT         NOT NULL,
  query         TEXT         NOT NULL,
  doc_type      VARCHAR(20)  NOT NULL DEFAULT 'policy',
  article_count INT          NOT NULL DEFAULT 0,
  topic_plan    TEXT         NOT NULL DEFAULT '',
  html_content  TEXT         NOT NULL,
  created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS journal_reports_user_id_idx    ON journal_reports(user_id);
CREATE INDEX IF NOT EXISTS journal_reports_created_at_idx ON journal_reports(created_at DESC);

DROP TRIGGER IF EXISTS journal_reports_updated_at ON journal_reports;
CREATE TRIGGER journal_reports_updated_at
  BEFORE UPDATE ON journal_reports
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();


INSERT INTO accounts (name, email, password_hash, role, status)
VALUES (
  'Admin',
  'musya@gmail.com',
  crypt('123456musya', gen_salt('bf', 12)),
  'admin',
  'approved'
);
INSERT INTO accounts (name, email, password_hash, role, status)
VALUES (
  'Adminsuper',
  'supermusya@gmail.com',
  crypt('123456musya', gen_salt('bf', 12)),
  'adminsuper',
  'approved'
);