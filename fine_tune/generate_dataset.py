#!/usr/bin/env python3
import json
import random
import sqlite3
from pathlib import Path
import re

SCHEMA_PROMPT = """You are Drona, an elite T20 cricket tactical AI agent.
You have access to a SQLite database. Only use the tables and columns listed below.
NEVER invent column names. NEVER guess table names.

DATABASE SCHEMA:

TABLE matchups_1v1:
  batter TEXT, bowler TEXT, balls_faced INT, runs_scored INT,
  dismissals INT, strike_rate REAL, average REAL, dot_pct REAL,
  boundary_pct REAL, fours INT, sixes INT,
  dismissal_rate_wilson_lower REAL

TABLE bowler_phase_stats:
  bowler TEXT, phase TEXT,  -- EXACT values: 'powerplay' | 'middle' | 'death'
  balls_bowled INT, runs_conceded INT, wickets INT,
  economy REAL, bowling_sr REAL, average REAL,
  dot_pct REAL, boundary_concede_pct REAL,
  fours_conceded INT, sixes_conceded INT,
  bowling_archetype TEXT  -- EXACT: 'death_specialist' | 'powerplay_enforcer' |
                          --        'strike_bowler' | 'economy_merchant' | 'allrounder'

TABLE batter_phase_stats:
  batter TEXT, phase TEXT,  -- EXACT values: 'powerplay' | 'middle' | 'death'
  balls_faced INT, runs_scored INT, dismissals INT,
  strike_rate REAL, average REAL, dot_pct REAL,
  boundary_pct REAL, fours INT, sixes INT,
  dismissal_rate_wilson_lower REAL

TABLE venue_profiles:
  venue TEXT, matches_played INT,
  avg_first_innings_score REAL, avg_second_innings_score REAL,
  avg_powerplay_runs REAL, avg_middle_runs REAL, avg_death_runs REAL,
  surface_behavior_index REAL

TABLE batter_vs_bowling_style:
  batter TEXT, bowling_style TEXT,  -- EXACT: 'pace' | 'spin' | 'unknown'
  balls_faced INT, runs_scored INT, dismissals INT,
  strike_rate REAL, average REAL, dot_pct REAL, boundary_pct REAL,
  fours INT, sixes INT

TABLE recent_form:
  player TEXT, role TEXT,  -- EXACT: 'batter' | 'bowler'
  recent_avg_runs REAL, recent_avg_sr REAL, recent_dismissal_rate REAL,
  recent_economy REAL, recent_wickets_per_match REAL,
  form_trend TEXT,  -- EXACT: 'improving' | 'declining' | 'stable'
  sample_matches INT

TABLE player_clones:
  player TEXT, clone TEXT, similarity REAL,
  clone_rank INT, role TEXT  -- EXACT: 'batter' | 'bowler'

TABLE pressure_performance:
  player TEXT, role TEXT,
  balls_faced INT, runs_scored REAL, dismissals INT,
  strike_rate REAL, dot_pct REAL, boundary_pct REAL,
  dismissal_rate REAL, fours INT, sixes INT,
  pressure_scenario TEXT

SQL RULES:
1. ALWAYS use LIKE '%name%' for player/venue name matching.
2. ALWAYS add LIMIT 10 unless you need all rows.
3. phase values are EXACTLY: 'powerplay', 'middle', 'death'
4. bowling_archetype values are EXACTLY as listed above.
5. NEVER use JOIN unless tables share a key column.
6. NEVER invent columns not in the schema above.

EXAMPLES:
Q: How does Kohli bat in death overs?
SQL: SELECT batter, phase, balls_faced, strike_rate, average, dot_pct, boundary_pct FROM batter_phase_stats WHERE batter LIKE '%Kohli%' AND phase = 'death' LIMIT 5;

Q: Best economy bowlers for powerplay?
SQL: SELECT bowler, economy, dot_pct, wickets, bowling_archetype FROM bowler_phase_stats WHERE phase = 'powerplay' ORDER BY economy ASC LIMIT 8;

Q: Bumrah vs Rohit head to head?
SQL: SELECT batter, bowler, balls_faced, runs_scored, dismissals, strike_rate, dot_pct, dismissal_rate_wilson_lower FROM matchups_1v1 WHERE batter LIKE '%Rohit%' AND bowler LIKE '%Bumrah%' LIMIT 5;

Q: Who are the death specialists?
SQL: SELECT bowler, economy, dot_pct, wickets, bowling_archetype FROM bowler_phase_stats WHERE phase = 'death' AND bowling_archetype = 'death_specialist' ORDER BY economy ASC LIMIT 6;

Q: Venue profile for Wankhede?
SQL: SELECT venue, matches_played, avg_first_innings_score, surface_behavior_index, avg_death_runs FROM venue_profiles WHERE venue LIKE '%Wankhede%' LIMIT 3;"""

COACH_SYS = "You are Drona, a gritty T20 head coach. Given verified stats, give ONE sharp tactical recommendation in 2-3 sentences. Never invent statistics. Never hedge."

batters = ["V Kohli", "RG Sharma", "MS Dhoni", "AB de Villiers", "SR Watson", "DA Warner", "S Dhawan", "CH Gayle", "KL Rahul", "AT Rayudu", "Yuvraj Singh", "SK Raina", "AM Rahane", "G Gambhir", "RV Uthappa"]
bowlers = ["JJ Bumrah", "SP Narine", "YS Chahal", "R Ashwin", "RA Jadeja", "A Mishra", "DJ Bravo", "SL Malinga", "B Kumar", "A Nehra", "Rashid Khan", "PP Chawla", "TS Mills", "MM Sharma", "KH Pandya"]
venues = ["Eden Gardens", "Wankhede Stadium", "M Chinnaswamy Stadium", "Feroz Shah Kotla", "MA Chidambaram Stadium", "DY Patil Stadium"]
phases = ["powerplay", "middle", "death"]
archetypes = ["death_specialist", "powerplay_enforcer", "strike_bowler", "economy_merchant", "allrounder"]
styles = ["pace", "spin"]

examples = []

# 25 matchups_1v1 queries
for _ in range(25):
    b1 = random.choice(batters)
    b2 = random.choice(bowlers)
    q = f"How has {b1} played against {b2}?"
    sql = f"SELECT batter, bowler, balls_faced, runs_scored, dismissals, strike_rate, dot_pct, dismissal_rate_wilson_lower FROM matchups_1v1 WHERE batter LIKE '%{b1.split()[-1]}%' AND bowler LIKE '%{b2.split()[-1]}%' LIMIT 5;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Query matchups_1v1 for {b1} against {b2}.\nSQL: {sql}"}]})

# 20 bowler_phase_stats (min 8 with archetype filter)
for i in range(20):
    if i < 8:
        ph = random.choice(phases)
        ar = random.choice(archetypes)
        q = f"Show me {ar}s in the {ph} phase."
        sql = f"SELECT bowler, economy, dot_pct, wickets, bowling_archetype FROM bowler_phase_stats WHERE phase = '{ph}' AND bowling_archetype = '{ar}' ORDER BY economy ASC LIMIT 6;"
        examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Filter bowler_phase_stats by phase and archetype.\nSQL: {sql}"}]})
    else:
        b2 = random.choice(bowlers)
        ph = random.choice(phases)
        q = f"What are {b2}'s stats in the {ph} overs?"
        sql = f"SELECT bowler, phase, economy, dot_pct, wickets, bowling_sr, bowling_archetype FROM bowler_phase_stats WHERE bowler LIKE '%{b2.split()[-1]}%' AND phase = '{ph}' LIMIT 5;"
        examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Need {ph}-over stats for {b2} from bowler_phase_stats.\nSQL: {sql}"}]})

# 20 batter_phase_stats
for _ in range(20):
    b1 = random.choice(batters)
    ph = random.choice(phases)
    q = f"How does {b1} perform in {ph} overs?"
    sql = f"SELECT batter, phase, balls_faced, strike_rate, average, dot_pct, boundary_pct FROM batter_phase_stats WHERE batter LIKE '%{b1.split()[-1]}%' AND phase = '{ph}' LIMIT 5;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Get {b1}'s performance in the {ph} phase.\nSQL: {sql}"}]})

# 15 venue_profiles
for _ in range(15):
    v = random.choice(venues)
    q = f"Give me the venue profile for {v}."
    v_clean = v.replace(" Stadium", "")
    sql = f"SELECT venue, matches_played, avg_first_innings_score, surface_behavior_index, avg_powerplay_runs, avg_middle_runs, avg_death_runs FROM venue_profiles WHERE venue LIKE '%{v_clean}%' LIMIT 3;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Lookup venue_profiles for {v}.\nSQL: {sql}"}]})

# 15 pressure_performance
for _ in range(15):
    b1 = random.choice(batters + bowlers)
    q = f"How does {b1} perform under pressure?"
    sql = f"SELECT player, role, pressure_scenario, balls_faced, runs_scored, strike_rate, dot_pct FROM pressure_performance WHERE player LIKE '%{b1.split()[-1]}%' LIMIT 5;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Check pressure_performance for {b1}.\nSQL: {sql}"}]})

# 10 recent_form
for _ in range(10):
    b1 = random.choice(batters + bowlers)
    q = f"What is the recent form of {b1}?"
    sql = f"SELECT player, role, recent_avg_runs, recent_avg_sr, recent_economy, form_trend FROM recent_form WHERE player LIKE '%{b1.split()[-1]}%' LIMIT 5;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Query recent_form for {b1}.\nSQL: {sql}"}]})

# 10 player_clones
for _ in range(10):
    b1 = random.choice(batters + bowlers)
    q = f"Find statistical clones for {b1}."
    sql = f"SELECT player, clone, similarity, clone_rank, role FROM player_clones WHERE player LIKE '%{b1.split()[-1]}%' ORDER BY similarity DESC LIMIT 3;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: Use player_clones table for {b1}.\nSQL: {sql}"}]})

# 5 multi-table queries (just standard SQL)
for _ in range(5):
    b1 = random.choice(batters)
    q = f"What is {b1}'s overall recent form and strike rate against spin?"
    sql = f"SELECT player, recent_avg_runs, recent_avg_sr FROM recent_form WHERE player LIKE '%{b1.split()[-1]}%' LIMIT 5;"
    examples.append({"messages": [{"role":"system","content":SCHEMA_PROMPT},{"role":"user","content":q},{"role":"assistant","content":f"THOUGHT: I'll start by checking recent_form for {b1}.\nSQL: {sql}"}]})

# 10 error recovery examples
for _ in range(10):
    b1 = random.choice(batters)
    q = f"What is {b1}'s batting average?"
    sql_bad = f"SELECT batter, phase, batting_average FROM batter_phase_stats WHERE batter LIKE '%{b1.split()[-1]}%';"
    sql_good = f"SELECT batter, phase, average, balls_faced, strike_rate FROM batter_phase_stats WHERE batter LIKE '%{b1.split()[-1]}%' LIMIT 5;"
    examples.append({"messages": [
        {"role":"system","content":SCHEMA_PROMPT},
        {"role":"user","content":q},
        {"role":"assistant","content":f"THOUGHT: Query batter_phase_stats for {b1}.\nSQL: {sql_bad}"},
        {"role":"user","content":"ERROR: Schema guard rejected query. Unknown column: 'batting_average'. Correction: Do NOT use 'batting_average'. The correct column is 'average'."},
        {"role":"assistant","content":f"THOUGHT: I used the wrong column name. The correct column is 'average', not 'batting_average'.\nSQL: {sql_good}"}
    ]})

# 20 coach narration examples
for _ in range(20):
    b2 = random.choice(bowlers)
    q = f"SITUATION: Over 17, defending 34 off 18 balls, Wankhede Stadium\nRESULTS: bowler: {b2}, phase: death, economy: 6.8, dot_pct: 44.2, wickets: 23, bowling_archetype: death_specialist\nCONFIDENCE: HIGH CONFIDENCE (n=156 balls, Wilson CI lower: 0.412)"
    examples.append({"messages": [
        {"role":"system","content":COACH_SYS},
        {"role":"user","content":q},
        {"role":"assistant","content":f"Bowl {b2} now and don't rotate him out — 6.8 economy with 44% dots in death overs is the best weapon you have. His Wilson CI is tight at 0.41, so trust it. Only risk: if the batter has seen him twice this innings, set a deceptive field before he bowls."}
    ]})

random.shuffle(examples)

train = examples[:120]
val = examples[120:]

Path("fine_tune").mkdir(exist_ok=True)
with open("fine_tune/train.jsonl", "w") as f:
    for e in train:
        f.write(json.dumps(e) + "\n")
with open("fine_tune/val.jsonl", "w") as f:
    for e in val:
        f.write(json.dumps(e) + "\n")

# Validation
conn = sqlite3.connect("db/cricket_drona.db")
errors, valid = 0, 0
with open("fine_tune/train.jsonl") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        last = [m for m in ex["messages"] if m["role"] == "assistant"][-1]
        if "SQL:" in last["content"]:
            sql = last["content"].split("SQL:")[-1].strip().rstrip(";") + ";"
            try:
                conn.execute(sql).fetchall()
                valid += 1
            except Exception as e:
                print(f"  Line {i+1} INVALID SQL: {e}")
                print(f"  SQL: {sql[:100]}")
                errors += 1
        else:
            valid += 1 # Coach narration
with open("fine_tune/val.jsonl") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        last = [m for m in ex["messages"] if m["role"] == "assistant"][-1]
        if "SQL:" in last["content"]:
            sql = last["content"].split("SQL:")[-1].strip().rstrip(";") + ";"
            try:
                conn.execute(sql).fetchall()
                valid += 1
            except Exception as e:
                print(f"  Line {i+1} INVALID SQL: {e}")
                print(f"  SQL: {sql[:100]}")
                errors += 1
        else:
            valid += 1 # Coach narration

print(f"\nTotal: {valid} valid, {errors} invalid")
conn.close()
