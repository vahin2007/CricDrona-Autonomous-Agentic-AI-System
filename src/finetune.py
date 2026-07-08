#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT DRONA — FINE-TUNING SCRIPT                              ║
║  Generates 550 training examples then runs LoRA fine-tuning      ║
║                                                                  ║
║  Usage:                                                          ║
║    python src/finetune.py              (generate + train)        ║
║    python src/finetune.py --gen-only   (only generate dataset)   ║
║    python src/finetune.py --validate   (validate existing JSONL) ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
DB_PATH      = Path("db/cricket_drona.db")
TRAIN_PATH   = Path("fine_tune/train.jsonl")
VAL_PATH     = Path("fine_tune/val.jsonl")
ADAPTER_DIR  = Path("models/drona-lora")
GGUF_PATH    = Path("models/drona-lora.gguf")
MODELFILE    = Path("models/Modelfile")

# ─────────────────────────────────────────────────────────────────
# SCHEMA PROMPT — injected into every SQL generation example
# ─────────────────────────────────────────────────────────────────
SCHEMA = """You are Drona, an elite T20 cricket tactical AI agent.
You have access to a SQLite database. Use ONLY the tables and columns listed.
NEVER invent column names. NEVER treat tool names as SQL tables.
memory_read, memory_write, clone_lookup, momentum_detector are TOOLS — NOT tables.

DATABASE SCHEMA:

TABLE matchups_1v1:
  batter TEXT, bowler TEXT, balls_faced INT, runs_scored INT,
  dismissals INT, strike_rate REAL, average REAL, dot_pct REAL,
  boundary_pct REAL, fours INT, sixes INT,
  dismissal_rate_wilson_lower REAL

TABLE bowler_phase_stats:
  bowler TEXT, phase TEXT, balls_bowled INT, runs_conceded INT,
  wickets INT, economy REAL, bowling_sr REAL, average REAL,
  dot_pct REAL, boundary_concede_pct REAL,
  fours_conceded INT, sixes_conceded INT, bowling_archetype TEXT
  -- phase EXACT: 'powerplay' | 'middle' | 'death'
  -- bowling_archetype EXACT: 'death_specialist' | 'powerplay_enforcer' |
  --   'strike_bowler' | 'economy_merchant' | 'allrounder'

TABLE batter_phase_stats:
  batter TEXT, phase TEXT, balls_faced INT, runs_scored INT,
  dismissals INT, strike_rate REAL, average REAL, dot_pct REAL,
  boundary_pct REAL, fours INT, sixes INT,
  dismissal_rate_wilson_lower REAL

TABLE venue_profiles:
  venue TEXT, matches_played INT, avg_first_innings_score REAL,
  avg_second_innings_score REAL, avg_powerplay_runs REAL,
  avg_middle_runs REAL, avg_death_runs REAL, surface_behavior_index REAL
  -- surface_behavior_index < 1.0 = slow/damp, > 1.0 = flat/batting friendly

TABLE batter_vs_bowling_style:
  batter TEXT, bowling_style TEXT, balls_faced INT, runs_scored INT,
  dismissals INT, strike_rate REAL, average REAL, dot_pct REAL,
  boundary_pct REAL, fours INT, sixes INT
  -- bowling_style EXACT: 'pace' | 'spin' | 'unknown'

TABLE recent_form:
  player TEXT, role TEXT, recent_avg_runs REAL, recent_avg_sr REAL,
  recent_dismissal_rate REAL, recent_economy REAL,
  recent_wickets_per_match REAL, form_trend TEXT, sample_matches INT
  -- role EXACT: 'batter' | 'bowler'
  -- form_trend EXACT: 'improving' | 'declining' | 'stable'

TABLE player_clones:
  player TEXT, clone TEXT, similarity REAL, clone_rank INT, role TEXT

TABLE pressure_performance:
  player TEXT, role TEXT, balls_faced INT, runs_scored REAL,
  dismissals INT, strike_rate REAL, dot_pct REAL, boundary_pct REAL,
  dismissal_rate REAL, fours INT, sixes INT, pressure_scenario TEXT

SQL RULES — NEVER BREAK:
1. ALWAYS use LIKE '%name%' for player/venue matching. NEVER exact =.
2. ALWAYS add LIMIT 10 unless you need all rows.
3. phase values EXACTLY: 'powerplay', 'middle', 'death'
4. bowling_archetype EXACTLY as listed above.
5. NEVER JOIN unless tables share a key.
6. NEVER invent columns.
7. memory_read is a TOOL — never query it as a table.
8. For uncapped players: use clone_lookup TOOL, not sql_executor."""

COACH_SYS = """You are Drona, a gritty T20 head coach.
The data analyst has queried the database. Numbers below are verified facts.
Give ONE sharp tactical recommendation based purely on these numbers.
Do NOT invent statistics. Do NOT hedge. Maximum 3 sentences.
Start with the recommendation. Cite the key number. Add one risk if confidence is low."""

# ─────────────────────────────────────────────────────────────────
# REAL PLAYER NAMES (verified in DB via ETL)
# ─────────────────────────────────────────────────────────────────
BATTERS = [
    "V Kohli", "RG Sharma", "DA Warner", "MS Dhoni", "AB de Villiers",
    "CH Gayle", "SR Watson", "SK Raina", "G Gambhir", "S Dhawan",
    "KL Rahul", "AT Rayudu", "AM Rahane", "RV Uthappa", "Yuvraj Singh",
    "JP Duminy", "SS Iyer", "MA Agarwal", "SV Samson", "HH Pandya",
    "KA Pollard", "DJ Bravo", "AD Russell", "SP Narine", "SN Khan",
    "PP Shaw", "RA Tripathi", "Q de Kock", "F du Plessis", "JC Buttler",
]

BOWLERS = [
    "JJ Bumrah", "YS Chahal", "R Ashwin", "RA Jadeja", "A Mishra",
    "SP Narine", "Rashid Khan", "B Kumar", "MM Sharma", "TS Mills",
    "A Nehra", "PP Chawla", "DJ Bravo", "SL Malinga", "KH Pandya",
    "DL Chahar", "Harbhajan Singh", "Z Khan", "P Kumar", "S Gopal",
    "T Natarajan", "Arshdeep Singh", "HV Patel", "M Pathirana",
    "W Sundar", "K Rabada", "C Ganapathi", "PM Krishna", "SK Warne",
]

VENUES = [
    "Eden Gardens", "Wankhede Stadium", "M Chinnaswamy Stadium",
    "Feroz Shah Kotla", "MA Chidambaram Stadium", "DY Patil Stadium",
    "Rajiv Gandhi International Stadium", "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium", "Dr DY Patil Sports Academy",
]

ARCHETYPES = [
    "death_specialist", "powerplay_enforcer",
    "strike_bowler", "economy_merchant", "allrounder",
]

PHASES = ["powerplay", "middle", "death"]
STYLES = ["pace", "spin"]
TRENDS = ["improving", "declining", "stable"]

# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────

def ex(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def rb() -> str: return random.choice(BATTERS)
def rbo() -> str: return random.choice(BOWLERS)
def rv() -> str: return random.choice(VENUES)
def rp() -> str: return random.choice(PHASES)
def ra() -> str: return random.choice(ARCHETYPES)
def rs() -> str: return random.choice(STYLES)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE GENERATORS — 11 categories, 550 total
# ─────────────────────────────────────────────────────────────────

def gen_matchups(n: int = 80) -> list[dict]:
    """Head-to-head batter vs bowler queries."""
    out = []
    for _ in range(n):
        b, bo = rb(), rbo()
        out.append(ex(SCHEMA,
            f"Head to head stats for {b} vs {bo}",
            f"THOUGHT: Need matchup data from matchups_1v1 for {b} against {bo}.\n"
            f"SQL: SELECT batter, bowler, balls_faced, runs_scored, dismissals, "
            f"strike_rate, dot_pct, dismissal_rate_wilson_lower "
            f"FROM matchups_1v1 WHERE batter LIKE '%{b.split()[-1]}%' "
            f"AND bowler LIKE '%{bo.split()[-1]}%' LIMIT 5;"
        ))
    for _ in range(n // 2):
        b = rb()
        out.append(ex(SCHEMA,
            f"Who has dismissed {b} the most times?",
            f"THOUGHT: Find bowlers with most dismissals against {b} in matchups_1v1.\n"
            f"SQL: SELECT bowler, dismissals, balls_faced, strike_rate, dot_pct "
            f"FROM matchups_1v1 WHERE batter LIKE '%{b.split()[-1]}%' "
            f"ORDER BY dismissals DESC LIMIT 8;"
        ))
    for _ in range(n // 2):
        bo = rbo()
        out.append(ex(SCHEMA,
            f"Which batters does {bo} struggle against?",
            f"THOUGHT: Find batters with highest strike rate against {bo} — high SR means bowler struggles.\n"
            f"SQL: SELECT batter, bowler, balls_faced, runs_scored, strike_rate, dismissals "
            f"FROM matchups_1v1 WHERE bowler LIKE '%{bo.split()[-1]}%' "
            f"AND balls_faced >= 10 ORDER BY strike_rate DESC LIMIT 8;"
        ))
    return out


def gen_bowler_phase(n: int = 80) -> list[dict]:
    """Bowler phase stats including archetype queries."""
    out = []
    for _ in range(n):
        bo, p = rbo(), rp()
        out.append(ex(SCHEMA,
            f"How does {bo} bowl in {p} overs?",
            f"THOUGHT: Query bowler_phase_stats for {bo} in {p} phase.\n"
            f"SQL: SELECT bowler, phase, economy, dot_pct, wickets, bowling_sr, bowling_archetype "
            f"FROM bowler_phase_stats WHERE bowler LIKE '%{bo.split()[-1]}%' "
            f"AND phase = '{p}' LIMIT 5;"
        ))
    for _ in range(n // 2):
        a, p = ra(), rp()
        out.append(ex(SCHEMA,
            f"Who are the best {a.replace('_',' ')}s in the squad?",
            f"THOUGHT: Filter bowler_phase_stats by bowling_archetype = '{a}' for {p} phase.\n"
            f"SQL: SELECT bowler, economy, dot_pct, wickets, bowling_archetype "
            f"FROM bowler_phase_stats WHERE bowling_archetype = '{a}' "
            f"AND phase = '{p}' ORDER BY economy ASC LIMIT 8;"
        ))
    for _ in range(n // 3):
        p = rp()
        out.append(ex(SCHEMA,
            f"Top 8 most economical bowlers in {p} overs",
            f"THOUGHT: Sort bowler_phase_stats by economy ascending for {p} phase.\n"
            f"SQL: SELECT bowler, economy, dot_pct, wickets, balls_bowled, bowling_archetype "
            f"FROM bowler_phase_stats WHERE phase = '{p}' "
            f"ORDER BY economy ASC LIMIT 8;"
        ))
    # Archetype-specific — critical for fixing base model failures
    for a in ARCHETYPES:
        for p in PHASES:
            out.append(ex(SCHEMA,
                f"List all {a.replace('_',' ')} bowlers for {p} overs",
                f"THOUGHT: The bowling_archetype column has exact value '{a}'. Query bowler_phase_stats.\n"
                f"SQL: SELECT bowler, economy, dot_pct, wickets, bowling_sr "
                f"FROM bowler_phase_stats WHERE bowling_archetype = '{a}' "
                f"AND phase = '{p}' ORDER BY economy ASC LIMIT 8;"
            ))
    return out


def gen_batter_phase(n: int = 70) -> list[dict]:
    """Batter phase stats queries."""
    out = []
    for _ in range(n):
        b, p = rb(), rp()
        out.append(ex(SCHEMA,
            f"How does {b} bat in {p} overs?",
            f"THOUGHT: Query batter_phase_stats for {b} in {p} phase.\n"
            f"SQL: SELECT batter, phase, balls_faced, strike_rate, average, "
            f"dot_pct, boundary_pct FROM batter_phase_stats "
            f"WHERE batter LIKE '%{b.split()[-1]}%' AND phase = '{p}' LIMIT 5;"
        ))
    for _ in range(n // 2):
        p = rp()
        out.append(ex(SCHEMA,
            f"Top 10 batters by strike rate in {p} overs",
            f"THOUGHT: Sort batter_phase_stats by strike_rate descending for {p}.\n"
            f"SQL: SELECT batter, phase, balls_faced, strike_rate, average, boundary_pct "
            f"FROM batter_phase_stats WHERE phase = '{p}' AND balls_faced >= 30 "
            f"ORDER BY strike_rate DESC LIMIT 10;"
        ))
    for _ in range(n // 3):
        b = rb()
        out.append(ex(SCHEMA,
            f"Full phase breakdown for {b}",
            f"THOUGHT: Get all three phases for {b} from batter_phase_stats.\n"
            f"SQL: SELECT batter, phase, balls_faced, strike_rate, average, dot_pct, boundary_pct "
            f"FROM batter_phase_stats WHERE batter LIKE '%{b.split()[-1]}%' "
            f"ORDER BY phase LIMIT 10;"
        ))
    return out


def gen_venue(n: int = 50) -> list[dict]:
    """Venue profile queries."""
    out = []
    for _ in range(n):
        v = rv()
        vshort = v.split()[0]
        out.append(ex(SCHEMA,
            f"What is the venue profile for {v}?",
            f"THOUGHT: Query venue_profiles for {v} — check avg score and SBI.\n"
            f"SQL: SELECT venue, matches_played, avg_first_innings_score, "
            f"surface_behavior_index, avg_powerplay_runs, avg_death_runs "
            f"FROM venue_profiles WHERE venue LIKE '%{vshort}%' LIMIT 3;"
        ))
    out.append(ex(SCHEMA,
        "Which venue has the slowest pitch (lowest SBI)?",
        "THOUGHT: Sort venue_profiles by surface_behavior_index ascending.\n"
        "SQL: SELECT venue, surface_behavior_index, avg_first_innings_score, matches_played "
        "FROM venue_profiles ORDER BY surface_behavior_index ASC LIMIT 5;"
    ))
    out.append(ex(SCHEMA,
        "Best batting venues in IPL by average first innings score",
        "THOUGHT: Sort venue_profiles by avg_first_innings_score descending.\n"
        "SQL: SELECT venue, avg_first_innings_score, surface_behavior_index, matches_played "
        "FROM venue_profiles ORDER BY avg_first_innings_score DESC LIMIT 8;"
    ))
    out.append(ex(SCHEMA,
        "Is it better to bat first or chase at Eden Gardens?",
        "THOUGHT: Compare avg_first_innings_score vs avg_second_innings_score at Eden Gardens.\n"
        "SQL: SELECT venue, avg_first_innings_score, avg_second_innings_score, "
        "surface_behavior_index FROM venue_profiles WHERE venue LIKE '%Eden%' LIMIT 3;"
    ))
    return out


def gen_batter_vs_style(n: int = 50) -> list[dict]:
    """Batter performance vs bowling style."""
    out = []
    for _ in range(n):
        b, s = rb(), rs()
        out.append(ex(SCHEMA,
            f"How does {b} play against {s} bowling?",
            f"THOUGHT: Query batter_vs_bowling_style for {b} against {s}.\n"
            f"SQL: SELECT batter, bowling_style, balls_faced, strike_rate, average, dot_pct, boundary_pct "
            f"FROM batter_vs_bowling_style WHERE batter LIKE '%{b.split()[-1]}%' "
            f"AND bowling_style = '{s}' LIMIT 5;"
        ))
    for _ in range(n // 3):
        s = rs()
        out.append(ex(SCHEMA,
            f"Which batters struggle most against {s} — highest dot percentage?",
            f"THOUGHT: Find batters with high dot_pct against {s} bowling — they struggle.\n"
            f"SQL: SELECT batter, bowling_style, balls_faced, dot_pct, strike_rate, average "
            f"FROM batter_vs_bowling_style WHERE bowling_style = '{s}' AND balls_faced >= 20 "
            f"ORDER BY dot_pct DESC LIMIT 10;"
        ))
    return out


def gen_recent_form(n: int = 50) -> list[dict]:
    """Recent form queries."""
    out = []
    for _ in range(n // 2):
        p = random.choice(BATTERS + BOWLERS)
        role = "batter" if p in BATTERS else "bowler"
        out.append(ex(SCHEMA,
            f"What is {p}'s recent form?",
            f"THOUGHT: Query recent_form for {p} to check trend.\n"
            f"SQL: SELECT player, role, recent_avg_runs, recent_avg_sr, recent_economy, "
            f"recent_wickets_per_match, form_trend, sample_matches "
            f"FROM recent_form WHERE player LIKE '%{p.split()[-1]}%' LIMIT 5;"
        ))
    for t in TRENDS:
        out.append(ex(SCHEMA,
            f"Which bowlers are {t} in recent matches?",
            f"THOUGHT: Filter recent_form by form_trend = '{t}' and role = 'bowler'.\n"
            f"SQL: SELECT player, role, recent_economy, recent_wickets_per_match, form_trend "
            f"FROM recent_form WHERE role = 'bowler' AND form_trend = '{t}' "
            f"ORDER BY recent_economy ASC LIMIT 10;"
        ))
    for t in TRENDS:
        out.append(ex(SCHEMA,
            f"Top improving batters right now",
            f"THOUGHT: Filter recent_form by role = 'batter' and form_trend = 'improving'.\n"
            f"SQL: SELECT player, role, recent_avg_runs, recent_avg_sr, form_trend "
            f"FROM recent_form WHERE role = 'batter' AND form_trend = 'improving' "
            f"ORDER BY recent_avg_sr DESC LIMIT 10;"
        ))
    return out


def gen_pressure(n: int = 50) -> list[dict]:
    """Pressure performance queries."""
    out = []
    for _ in range(n // 2):
        p = random.choice(BATTERS + BOWLERS)
        out.append(ex(SCHEMA,
            f"How does {p} perform under pressure in death-over chases?",
            f"THOUGHT: Use pressure_performance table — separate from career stats.\n"
            f"SQL: SELECT player, role, balls_faced, strike_rate, dot_pct, "
            f"boundary_pct, dismissal_rate FROM pressure_performance "
            f"WHERE player LIKE '%{p.split()[-1]}%' LIMIT 5;"
        ))
    out.append(ex(SCHEMA,
        "Best death-over batters in pressure chases",
        "THOUGHT: Sort pressure_performance by strike_rate descending for batters.\n"
        "SQL: SELECT player, role, balls_faced, strike_rate, boundary_pct, dismissal_rate "
        "FROM pressure_performance WHERE role = 'batter' AND balls_faced >= 10 "
        "ORDER BY strike_rate DESC LIMIT 10;"
    ))
    out.append(ex(SCHEMA,
        "Most economical bowlers in pressure chases",
        "THOUGHT: Sort pressure_performance by strike_rate ascending for bowlers — lower is better.\n"
        "SQL: SELECT player, role, balls_faced, strike_rate, dot_pct, dismissal_rate "
        "FROM pressure_performance WHERE role = 'bowler' AND balls_faced >= 10 "
        "ORDER BY strike_rate ASC LIMIT 10;"
    ))
    for _ in range(n // 3):
        players = random.sample(BATTERS, 3)
        names_like = " OR ".join([f"player LIKE '%{p.split()[-1]}%'" for p in players])
        out.append(ex(SCHEMA,
            f"Compare pressure performance: {', '.join(players)}",
            f"THOUGHT: Query pressure_performance for multiple players to compare.\n"
            f"SQL: SELECT player, role, balls_faced, strike_rate, dot_pct, boundary_pct, dismissal_rate "
            f"FROM pressure_performance WHERE ({names_like}) ORDER BY strike_rate DESC LIMIT 10;"
        ))
    return out


def gen_clones(n: int = 40) -> list[dict]:
    """OOD clone lookup — CRITICAL: must use clone_lookup TOOL not sql_executor."""
    out = []
    uncapped = [
        "Arjun Tendulkar", "Yashasvi Jaiswal early career",
        "Prithvi Shaw rookie", "Tilak Varma new player",
        "Dewald Brevis unknown", "Tristan Stubbs uncapped",
        "Rajvardhan Hangargekar", "Kumar Kushagra", "Shaik Rasheed",
        "Nishant Sindhu", "Ricky Bhui", "Sai Kishore", "Akash Singh",
    ]
    for _ in range(n // 2):
        name = random.choice(uncapped).split()[0] + " " + random.choice(uncapped).split()[-1]
        out.append(ex(SCHEMA,
            f"{name} is playing but I have no data on them. What can Drona do?",
            f"THOUGHT: {name} is not in the database. I must use the clone_lookup TOOL — "
            f"NOT sql_executor — to find their statistical twin.\n"
            f"ACTION: clone_lookup\n"
            f"INPUT: {name.split()[0]}"
        ))
    for _ in range(n // 2):
        p = random.choice(BOWLERS)
        out.append(ex(SCHEMA,
            f"Find players statistically similar to {p}",
            f"THOUGHT: Use player_clones table to find statistical twins of {p}.\n"
            f"SQL: SELECT player, clone, similarity, clone_rank, role "
            f"FROM player_clones WHERE player LIKE '%{p.split()[-1]}%' "
            f"ORDER BY similarity DESC LIMIT 3;"
        ))
    # Critical: player not found → switch to clone_lookup
    for _ in range(n // 4):
        out.append(ex(SCHEMA,
            "The query returned zero rows for this player. What do I do?",
            "THOUGHT: Zero rows means the player is not in the database. "
            "I must NOT try sql_executor again. I must call clone_lookup TOOL "
            "with the player name to find their statistical twin.\n"
            "ACTION: clone_lookup\n"
            "INPUT: <player_name>"
        ))
    return out


def gen_memory_tool_correct(n: int = 50) -> list[dict]:
    """
    CRITICAL CATEGORY — fixes the #1 base model failure:
    model treats memory_read as a SQL table.
    Every example here teaches: memory_read is a TOOL, never a SQL table.
    """
    out = []
    # Explicitly teaching that memory_read is NOT a table
    out.append(ex(SCHEMA,
        "Check if we have any lessons for death overs at Wankhede",
        "THOUGHT: memory_read is a TOOL, not a SQL table. I must call it as a tool, "
        "NOT write SELECT FROM memory_read.\n"
        "ACTION: memory_read\n"
        "INPUT: Wankhede death overs"
    ))
    out.append(ex(SCHEMA,
        "What lessons do we have from past matches?",
        "THOUGHT: Past lessons are stored in episodic memory accessed via the memory_read TOOL. "
        "I cannot query it with SQL — it is NOT a database table.\n"
        "ACTION: memory_read\n"
        "INPUT: past match lessons"
    ))
    out.append(ex(SCHEMA,
        "SELECT lesson FROM memory_read LIMIT 5;",
        "THOUGHT: This SQL is WRONG. memory_read is a TOOL, NOT a table. "
        "There is no memory_read table in the database. "
        "The valid tables are: matchups_1v1, bowler_phase_stats, batter_phase_stats, "
        "venue_profiles, batter_vs_bowling_style, recent_form, player_clones, pressure_performance. "
        "To access past lessons I must call the memory_read TOOL.\n"
        "ACTION: memory_read\n"
        "INPUT: general lessons"
    ))
    # Error recovery — model got rejected for querying memory_read
    out.append(ex(SCHEMA,
        "Error: Unknown table 'memory_read'. Valid tables: matchups_1v1, bowler_phase_stats...",
        "THOUGHT: I made an error by treating memory_read as a SQL table. "
        "memory_read is a TOOL that I call with an action, not a SELECT statement. "
        "I will now call it correctly as a tool.\n"
        "ACTION: memory_read\n"
        "INPUT: relevant context"
    ))
    for _ in range(n - 4):
        v = rv()
        p = rp()
        out.append(ex(SCHEMA,
            f"Do we have any coaching lessons for {v} in {p} overs?",
            f"THOUGHT: Coaching lessons are in episodic memory. memory_read is a TOOL — "
            f"I call it as an action, NOT as a SQL table.\n"
            f"ACTION: memory_read\n"
            f"INPUT: {v} {p}"
        ))
    return out


def gen_error_recovery(n: int = 50) -> list[dict]:
    """Multi-turn error recovery examples."""
    out = []
    # Wrong column name recovery
    bad_cols = [
        ("batting_average", "average", "batter_phase_stats"),
        ("bowling_average", "average", "bowler_phase_stats"),
        ("strike_rate_career", "strike_rate", "matchups_1v1"),
        ("economy_rate", "economy", "bowler_phase_stats"),
        ("batting_team", "batter", "batter_phase_stats"),
        ("wickets_taken", "wickets", "bowler_phase_stats"),
        ("balls_bowled_total", "balls_bowled", "bowler_phase_stats"),
        ("dismissal_rate_wilson_lower", "dismissal_rate_wilson_lower", "matchups_1v1"),
        ("runs_per_over", "economy", "bowler_phase_stats"),
        ("batter_sr", "strike_rate", "batter_phase_stats"),
    ]
    for bad, correct, table in bad_cols:
        b = rb()
        out.append({
            "messages": [
                {"role": "system",    "content": SCHEMA},
                {"role": "user",      "content": f"What is {b}'s {bad.replace('_',' ')}?"},
                {"role": "assistant", "content":
                    f"THOUGHT: I need {bad.replace('_',' ')} from {table}.\n"
                    f"SQL: SELECT {bad} FROM {table} WHERE batter LIKE '%{b.split()[-1]}%' LIMIT 5;"},
                {"role": "user",      "content":
                    f"ERROR: Schema guard rejected query. Unknown column: '{bad}'. "
                    f"Correction: Do NOT use '{bad}'. The correct column is '{correct}'."},
                {"role": "assistant", "content":
                    f"THOUGHT: I used the wrong column name '{bad}'. "
                    f"The correct column is '{correct}' in {table}. Let me fix the query.\n"
                    f"SQL: SELECT {correct}, "
                    f"{'balls_bowled' if 'bowler' in table else 'balls_faced'} "
                    f"FROM {table} WHERE "
                    f"{'bowler' if 'bowler' in table else 'batter'} "
                    f"LIKE '%{b.split()[-1]}%' LIMIT 5;"},
            ]
        })
    # Wrong table name recovery
    for _ in range(n // 4):
        b, bo = rb(), rbo()
        out.append({
            "messages": [
                {"role": "system",    "content": SCHEMA},
                {"role": "user",      "content": f"Matchup between {b} and {bo}"},
                {"role": "assistant", "content":
                    f"THOUGHT: Need head-to-head from matchup_stats table.\n"
                    f"SQL: SELECT * FROM matchup_stats WHERE batter LIKE '%{b.split()[-1]}%' LIMIT 5;"},
                {"role": "user",      "content":
                    "ERROR: Unknown table 'matchup_stats'. "
                    "Valid tables: matchups_1v1, bowler_phase_stats, batter_phase_stats, "
                    "venue_profiles, batter_vs_bowling_style, recent_form, player_clones, pressure_performance"},
                {"role": "assistant", "content":
                    f"THOUGHT: The correct table name is 'matchups_1v1', not 'matchup_stats'. Let me fix this.\n"
                    f"SQL: SELECT batter, bowler, balls_faced, runs_scored, dismissals, "
                    f"strike_rate, dot_pct, dismissal_rate_wilson_lower "
                    f"FROM matchups_1v1 WHERE batter LIKE '%{b.split()[-1]}%' "
                    f"AND bowler LIKE '%{bo.split()[-1]}%' LIMIT 5;"},
            ]
        })
    return out


def gen_coach_narration(n: int = 80) -> list[dict]:
    """Coach narration — Pass 2 examples with locked verified stats."""
    out = []
    # Death over bowling recommendations
    for _ in range(n // 4):
        bo = rbo()
        v = rv()
        econ = round(random.uniform(6.0, 9.0), 2)
        dot = round(random.uniform(25, 55), 1)
        wkts = random.randint(8, 40)
        n_balls = random.randint(80, 300)
        arch = random.choice(["death_specialist", "strike_bowler"])
        conf = "HIGH CONFIDENCE" if n_balls >= 60 else "MODERATE CONFIDENCE"
        wci = round(random.uniform(0.05, 0.35), 3)
        out.append(ex(COACH_SYS,
            f"SITUATION: Over {random.randint(16,19)}, defending {random.randint(160,195)}, "
            f"{v}, need to stop the boundary\n"
            f"RESULTS: bowler: {bo}, phase: death, economy: {econ}, dot_pct: {dot}, "
            f"wickets: {wkts}, bowling_archetype: {arch}\n"
            f"CONFIDENCE: {conf} (n={n_balls} balls, Wilson CI lower: {wci})",
            f"Bowl {bo} now — {econ} economy with {dot}% dots in death overs "
            f"is your best option at {v}. "
            f"{'Trust it — ' + str(n_balls) + ' balls is a solid sample.' if n_balls >= 60 else 'Small sample risk — watch for any signs of off-form before committing the full quota.'}"
        ))
    # Batter matchup advice
    for _ in range(n // 4):
        b, bo = rb(), rbo()
        sr = round(random.uniform(80, 180), 1)
        dot = round(random.uniform(15, 55), 1)
        dism = random.randint(0, 8)
        balls = random.randint(15, 160)
        wci = round(random.uniform(0.01, 0.4), 3)
        conf = "HIGH CONFIDENCE" if balls >= 60 else "MODERATE CONFIDENCE" if balls >= 25 else "LOW CONFIDENCE"
        out.append(ex(COACH_SYS,
            f"SITUATION: {b} is on strike, {bo} is bowling, middle overs\n"
            f"RESULTS: batter: {b}, bowler: {bo}, balls_faced: {balls}, "
            f"strike_rate: {sr}, dismissals: {dism}, dot_pct: {dot}, "
            f"dismissal_rate_wilson_lower: {wci}\n"
            f"CONFIDENCE: {conf} (n={balls} balls)",
            f"{'Keep ' + bo + ' on — ' if dot > 35 else 'Consider a bowling change — '}"
            f"{dot}% dots {'shows dominance' if dot > 35 else 'is not enough pressure'} against {b}. "
            f"{'High confidence read at ' + str(balls) + ' balls faced.' if balls >= 60 else 'Only ' + str(balls) + ' balls — use with caution.'}"
        ))
    # Venue-based advice
    for _ in range(n // 4):
        v = rv()
        sbi = round(random.uniform(0.80, 1.20), 3)
        avg = round(random.uniform(150, 180), 1)
        out.append(ex(COACH_SYS,
            f"SITUATION: Pre-match planning at {v}, deciding bowling strategy\n"
            f"RESULTS: venue: {v}, avg_first_innings_score: {avg}, "
            f"surface_behavior_index: {sbi}, matches_played: {random.randint(40,90)}\n"
            f"CONFIDENCE: HIGH CONFIDENCE",
            f"{'Spin-heavy lineup — SBI of ' + str(sbi) + ' confirms a slow pitch' if sbi < 0.95 else 'Pace attack first — SBI of ' + str(sbi) + ' means flat track'} at {v}. "
            f"Average first innings is {avg} — plan your target around that. "
            f"{'Prioritise dot balls over wickets in the powerplay.' if sbi < 0.95 else 'Wicket-taking in the first 6 is critical on this surface.'}"
        ))
    # Pressure chase advice
    for _ in range(n // 4):
        p = random.choice(BATTERS)
        sr = round(random.uniform(120, 200), 1)
        dot = round(random.uniform(10, 40), 1)
        balls = random.randint(15, 80)
        conf = "HIGH CONFIDENCE" if balls >= 60 else "MODERATE CONFIDENCE" if balls >= 25 else "LOW CONFIDENCE"
        out.append(ex(COACH_SYS,
            f"SITUATION: Death over pressure chase, need 40 off 24\n"
            f"RESULTS: player: {p}, role: batter, balls_faced: {balls}, "
            f"strike_rate: {sr}, dot_pct: {dot}, boundary_pct: {round(random.uniform(20,45),1)}\n"
            f"CONFIDENCE: {conf} (n={balls} balls, pressure scenario: inn2_over16+)",
            f"Send {p} to bat — {sr} strike rate in pressure chases "
            f"{'is elite clutch performance' if sr >= 155 else 'gets the job done'}. "
            f"{'Only ' + str(balls) + ' pressure balls in sample — factor in current form too.' if balls < 25 else str(balls) + ' pressure balls is solid evidence.'}"
        ))
    return out


def gen_multi_step(n: int = 40) -> list[dict]:
    """Multi-step queries combining multiple tables."""
    out = []
    for _ in range(n // 2):
        b, bo, p = rb(), rbo(), rp()
        out.append(ex(SCHEMA,
            f"Should I bowl {bo} against {b} in {p} overs at {rv()}?",
            f"THOUGHT: I need two pieces of data: {bo}'s {p} stats AND {b} vs {bo} head-to-head. "
            f"Start with the most specific query — matchup data.\n"
            f"SQL: SELECT batter, bowler, balls_faced, strike_rate, dismissals, dot_pct "
            f"FROM matchups_1v1 WHERE batter LIKE '%{b.split()[-1]}%' "
            f"AND bowler LIKE '%{bo.split()[-1]}%' LIMIT 5;"
        ))
    for _ in range(n // 2):
        v, p = rv(), rp()
        s = rs()
        out.append(ex(SCHEMA,
            f"Is {v} a good venue for {s} bowling in {p} overs?",
            f"THOUGHT: Check venue profile for {v} first — SBI tells us pitch behavior. "
            f"Then check batter_vs_bowling_style for {s} bowling weakness.\n"
            f"SQL: SELECT venue, surface_behavior_index, avg_first_innings_score, "
            f"avg_{p}_runs FROM venue_profiles WHERE venue LIKE '%{v.split()[0]}%' LIMIT 3;"
        ))
    return out


# ─────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────

def build_dataset() -> tuple[list[dict], list[dict]]:
    random.seed(42)
    all_examples = []

    generators = [
        (gen_matchups,           80,  "matchups_1v1"),
        (gen_bowler_phase,       80,  "bowler_phase_stats"),
        (gen_batter_phase,       70,  "batter_phase_stats"),
        (gen_venue,              50,  "venue_profiles"),
        (gen_batter_vs_style,    50,  "batter_vs_bowling_style"),
        (gen_recent_form,        50,  "recent_form"),
        (gen_pressure,           50,  "pressure_performance"),
        (gen_clones,             40,  "player_clones"),
        (gen_memory_tool_correct,50,  "memory_tool"),
        (gen_error_recovery,     50,  "error_recovery"),
        (gen_coach_narration,    80,  "coach_narration"),
        (gen_multi_step,         40,  "multi_step"),
    ]

    for gen_fn, count, label in generators:
        examples = gen_fn(count)
        all_examples.extend(examples)
        print(f"  Generated {len(examples):3d} examples — {label}")

    print(f"\nTotal: {len(all_examples)} examples")

    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.8)
    return all_examples[:split], all_examples[split:]


# ─────────────────────────────────────────────────────────────────
# SQL VALIDATOR
# ─────────────────────────────────────────────────────────────────

def validate_dataset(train: list[dict], val: list[dict]) -> int:
    """Validate all SQL examples against the real DB. Returns error count."""
    if not DB_PATH.exists():
        print(f"WARNING: DB not found at {DB_PATH}, skipping SQL validation")
        return 0

    conn   = sqlite3.connect(DB_PATH)
    errors = 0
    total  = 0

    for examples in [train, val]:
        for i, ex_item in enumerate(examples):
            msgs = ex_item["messages"]
            # Get last assistant message
            last = [m for m in msgs if m["role"] == "assistant"][-1]["content"]
            if "SQL:" not in last:
                continue
            sql_raw = last.split("SQL:")[-1].strip().split("\n")[0].strip()
            # Skip if it contains obvious non-SQL
            if any(x in sql_raw for x in ["<", ">", "ACTION:", "INPUT:"]):
                continue
            sql = sql_raw.rstrip(";") + ";"
            total += 1
            try:
                conn.execute(sql).fetchall()
            except Exception as e:
                errors += 1
                if errors <= 10:  # only show first 10
                    print(f"  SQL ERROR in example {i}: {e}")
                    print(f"  SQL: {sql[:100]}")

    conn.close()
    valid = total - errors
    print(f"\nSQL Validation: {valid}/{total} valid ({errors} errors)")
    return errors


# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────

def save_dataset(train: list[dict], val: list[dict]) -> None:
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_PATH, "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    with open(VAL_PATH, "w") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    print(f"Saved: {TRAIN_PATH} ({len(train)} examples)")
    print(f"Saved: {VAL_PATH} ({len(val)} examples)")


# ─────────────────────────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────────────────────────

def run_finetune() -> None:
    import torch
    free_vram = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated()
    ) / 1e9
    print(f"VRAM free: {free_vram:.1f} GB")
    if free_vram < 5.0:
        print("ERROR: Need at least 5GB free VRAM. Run: pkill ollama && sleep 5")
        sys.exit(1)

    USE_UNSLOTH = False
    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
        print("Using Unsloth")
    except ImportError:
        print("Unsloth not found — using HuggingFace PEFT fallback")

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── Load model ─────────────────────────────────────────────
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = "unsloth/Meta-Llama-3.1-8B-Instruct",
            max_seq_length = 1024,
            load_in_4bit   = True,
            dtype          = None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r                          = 16,
            target_modules             = ["q_proj","k_proj","v_proj","o_proj",
                                          "gate_proj","up_proj","down_proj"],
            lora_alpha                 = 32,
            lora_dropout               = 0.00,
            bias                       = "none",
            use_gradient_checkpointing = "unsloth",
            random_state               = 42,
        )
    else:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
        from peft import LoraConfig, get_peft_model, TaskType

        bnb = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        model_id  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
        ))
        model.print_trainable_parameters()

    # ── Load dataset ───────────────────────────────────────────
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def fmt(ex_item):
        return {"text": tokenizer.apply_chat_template(
            ex_item["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = Dataset.from_list([fmt(e) for e in load_jsonl(TRAIN_PATH)])
    val_ds   = Dataset.from_list([fmt(e) for e in load_jsonl(VAL_PATH)])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Training args (tuned for 8GB Blackwell) ────────────────
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir                  = str(ADAPTER_DIR),
        num_train_epochs            = 3,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate               = 2e-4,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 20,
        
        # --- THE CHANGES ARE HERE ---
        save_strategy               = "no",
        eval_strategy               = "no",
        load_best_model_at_end      = False,
        # ----------------------------
        
        report_to                   = "none",
        gradient_checkpointing      = True,
        optim                       = "adamw_8bit" if USE_UNSLOTH else "paged_adamw_8bit",
        dataloader_num_workers      = 0,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        dataset_text_field = "text",
        max_seq_length     = 1024,
        args               = args,
    )

    print("\nStarting training... (Ctrl+C saves checkpoint)")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"\nAdapter saved: {ADAPTER_DIR}")

    # ── Modelfile ──────────────────────────────────────────────
    MODELFILE.parent.mkdir(exist_ok=True)
    MODELFILE.write_text(f"""FROM ./drona-lora.gguf
SYSTEM \"\"\"{SCHEMA}\"\"\"
PARAMETER temperature 0.1
PARAMETER num_predict 512
PARAMETER stop "<|eot_id|>"
""")
    print(f"Modelfile: {MODELFILE}")

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Convert to GGUF:
   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
   pip install -r ~/llama.cpp/requirements.txt
   python ~/llama.cpp/convert_hf_to_gguf.py models/drona-lora/ \\
     --outfile models/drona-lora.gguf --outtype q8_0

2. Register with Ollama:
   ollama serve &
   cd ~/drona/models && ollama create drona-v1 -f Modelfile

3. Test:
   cd ~/drona && python src/agent.py --test --model drona-v1

4. Compare:
   python src/evaluate.py --compare
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Drona — Fine-tuning")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only generate dataset, skip training")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing JSONL files")
    args = parser.parse_args()

    if args.validate:
        import json
        train = [json.loads(l) for l in open(TRAIN_PATH) if l.strip()]
        val   = [json.loads(l) for l in open(VAL_PATH)   if l.strip()]
        validate_dataset(train, val)
        return

    print("=" * 60)
    print("  PROJECT DRONA — DATASET GENERATION")
    print("=" * 60)
    train, val = build_dataset()
    errors = validate_dataset(train, val)
    save_dataset(train, val)

    print(f"\nDataset summary:")
    print(f"  Train: {len(train)} examples")
    print(f"  Val:   {len(val)} examples")
    print(f"  Total: {len(train)+len(val)} examples")
    print(f"  SQL errors: {errors}")

    if args.gen_only:
        print("\nDataset generated. Run without --gen-only to start training.")
        return

    if errors > 10:
        print(f"\nWARNING: {errors} SQL errors found. Review before training.")
        resp = input("Continue anyway? (y/N): ").strip().lower()
        if resp != "y":
            return

    print("\n" + "=" * 60)
    print("  PROJECT DRONA — FINE-TUNING")
    print("=" * 60)
    run_finetune()


if __name__ == "__main__":
    main()