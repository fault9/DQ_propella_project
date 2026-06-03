CREATE TABLE IF NOT EXISTS raters (
  name TEXT PRIMARY KEY,
  rubric_ack_at TEXT,
  first_seen_at TEXT,
  queue_built INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pair_assignments (
  rater_name TEXT,
  pair_id TEXT,
  queue_position INTEGER,
  side_assignment TEXT,
  PRIMARY KEY (rater_name, pair_id)
);

CREATE TABLE IF NOT EXISTS responses (
  response_id TEXT PRIMARY KEY,
  rater_name TEXT,
  pair_id TEXT,
  pair_kind TEXT,
  is_duplicate INTEGER,
  queue_position INTEGER,
  session_num INTEGER,
  side_assignment TEXT,
  readability_choice TEXT,
  substance_choice TEXT,
  skipped INTEGER DEFAULT 0,
  notes TEXT,
  time_ms INTEGER,
  submitted_at TEXT
);
