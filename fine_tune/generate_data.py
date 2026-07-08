import json
import sqlite3
import random
from pathlib import Path

DB_PATH = Path("db/cricket_drona.db")
TRAIN_PATH = Path("fine_tune/train.jsonl")
VAL_PATH = Path("fine_tune/val.jsonl")

# --- From src/agent.py ---
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
"""

COACH_SYSTEM_PROMPT = "You are Drona, a gritty T20 head coach. Given verified stats, give ONE sharp tactical recommendation in 2-3 sentences. Never invent statistics."

# Realistic IPL players
BATTERS = ["V Kohli", "RG Sharma", "MS Dhoni", "AB de Villiers", "SR Watson", "RA Jadeja", "DA Warner", "KL Rahul", "SK Raina"]
BOWLERS = ["JJ Bumrah", "YS Chahal", "R Ashwin", "SP Narine", "Rashid Khan", "SL Malinga", "B Kumar", "Mohammed Shami", "RA Jadeja"]
VENUES = ["Wankhede Stadium", "Eden Gardens", "M.Chinnaswamy Stadium", "MA Chidambaram Stadium", "Arun Jaitley Stadium"]
ARCHETYPES = ["death_specialist", "powerplay_enforcer", "strike_bowler", "economy_merchant", "allrounder"]
PHASES = ["powerplay", "middle", "death"]

def generate_sql_examples(count):
    examples = []
    
    # 20 examples: matchups_1v1
    for _ in range(20):
        b, bo = random.choice(BATTERS), random.choice(BOWLERS)
        examples.append({
            "user": f"Give me {bo} head to head stats against {b}",
            "assistant": f"THOUGHT: Need to fetch 1v1 matchup stats between {bo} and {b}.\nSQL: SELECT batter, bowler, balls_faced, runs_scored, dismissals, strike_rate, dismissal_rate_wilson_lower FROM matchups_1v1 WHERE batter LIKE '%{b}%' AND bowler LIKE '%{bo}%' LIMIT 10;"
        })

    # 15 examples: bowler_phase_stats
    for _ in range(15):
        bo, ph = random.choice(BOWLERS), random.choice(PHASES)
        examples.append({
            "user": f"Show {bo}'s performance in {ph} overs.",
            "assistant": f"THOUGHT: Looking for phase-specific stats for {bo} in {ph}.\nSQL: SELECT bowler, phase, balls_bowled, economy, wickets, dot_pct FROM bowler_phase_stats WHERE bowler LIKE '%{bo}%' AND phase = '{ph}' LIMIT 10;"
        })

    # 15 examples: batter_phase_stats
    for _ in range(15):
        ba, ph = random.choice(BATTERS), random.choice(PHASES)
        examples.append({
            "user": f"How does {ba} bat during the {ph} phase?",
            "assistant": f"THOUGHT: Querying batter performance for {ba} in the {ph} phase.\nSQL: SELECT batter, phase, balls_faced, strike_rate, average, boundary_pct FROM batter_phase_stats WHERE batter LIKE '%{ba}%' AND phase = '{ph}' LIMIT 10;"
        })

    # 10 examples: venue_profiles
    for _ in range(10):
        v = random.choice(VENUES)
        examples.append({
            "user": f"What is the venue profile for {v}?",
            "assistant": f"THOUGHT: Getting surface and scoring patterns for {v}.\nSQL: SELECT venue, avg_first_innings_score, surface_behavior_index, avg_powerplay_runs, avg_death_runs FROM venue_profiles WHERE venue LIKE '%{v}%' LIMIT 10;"
        })

    # 10 examples: pressure_performance
    for _ in range(10):
        p = random.choice(BATTERS + BOWLERS)
        examples.append({
            "user": f"How does {p} handle pressure situations?",
            "assistant": f"THOUGHT: Fetching pressure performance metrics for {p}.\nSQL: SELECT player, role, strike_rate, dismissal_rate, dot_pct FROM pressure_performance WHERE player LIKE '%{p}%' LIMIT 10;"
        })

    # 10 examples: bowling_archetype filter
    for arc in ARCHETYPES:
        for _ in range(2):
            examples.append({
                "user": f"Who are the top {arc.replace('_', ' ')} bowlers?",
                "assistant": f"THOUGHT: Filtering bowler phase stats by the {arc} archetype.\nSQL: SELECT bowler, economy, wickets, dot_pct FROM bowler_phase_stats WHERE bowling_archetype = '{arc}' ORDER BY economy ASC LIMIT 10;"
            })

    # 10 examples: recent_form
    for _ in range(10):
        p = random.choice(BATTERS + BOWLERS)
        examples.append({
            "user": f"What is {p}'s recent form trend?",
            "assistant": f"THOUGHT: Checking recent form and trend for {p}.\nSQL: SELECT player, form_trend, recent_avg_runs, recent_economy, sample_matches FROM recent_form WHERE player LIKE '%{p}%' LIMIT 10;"
        })

    # 5 examples: player_clones
    for _ in range(5):
        p = random.choice(BATTERS + BOWLERS)
        examples.append({
            "user": f"Find a statistical clone for {p}",
            "assistant": f"THOUGHT: Searching for players with similar statistical profiles to {p}.\nSQL: SELECT player, clone, similarity, role FROM player_clones WHERE player LIKE '%{p}%' LIMIT 10;"
        })

    # 5 examples: error recovery
    examples.extend([
        {
            "user": "Get economy for JJ Bumrah in death overs.",
            "assistant": "THOUGHT: Querying JJ Bumrah's death overs economy.\nSQL: SELECT economy FROM bowler_phase_stats WHERE bowler = 'JJ Bumrah' AND phase = 'death';"
        },
        {
            "user": "The previous SQL returned no rows because I used exact match. Try LIKE.",
            "assistant": "THOUGHT: Retrying with LIKE for fuzzy name matching.\nSQL: SELECT economy FROM bowler_phase_stats WHERE bowler LIKE '%Bumrah%' AND phase = 'death';"
        },
        {
            "user": "Select * from matches where id=1082591",
            "assistant": "THOUGHT: The user asked for a table not in my schema. I must use only schema tables.\nSQL: SELECT venue, avg_first_innings_score FROM venue_profiles LIMIT 10;"
        },
        {
            "user": "Who is the best batter?",
            "assistant": "THOUGHT: Ambiguous request, I will show top batters by average in all phases.\nSQL: SELECT batter, average, strike_rate FROM batter_phase_stats ORDER BY average DESC LIMIT 10;"
        },
        {
            "user": "Show me strike bowlers in powerplay",
            "assistant": "THOUGHT: Filtering powerplay enforcers/strike bowlers.\nSQL: SELECT bowler, economy, wickets FROM bowler_phase_stats WHERE phase = 'powerplay' AND bowling_archetype = 'strike_bowler' LIMIT 10;"
        }
    ])

    return examples[:count]

def generate_coach_examples(count):
    examples = []
    # 15: death over bowling
    for _ in range(15):
        bo = random.choice(BOWLERS)
        examples.append({
            "user": f"SITUATION: Over 19, defending 15 runs. {bo} is ready.\nRESULTS: economy: 8.5, wickets: 25, dot_pct: 35.0, phase: death\nCONFIDENCE: HIGH (n=450 balls)",
            "assistant": f"Bowl {bo} and target wide yorkers. His 8.5 death economy is elite, and a 35% dot ball rate will mount unbearable pressure. He has the temperament to defend this."
        })
    # 15: powerplay batting/bowling
    for _ in range(15):
        ba = random.choice(BATTERS)
        examples.append({
            "user": f"SITUATION: Over 3, chasing 180. {ba} at strike.\nRESULTS: strike_rate: 145.0, boundary_pct: 18.0, dot_pct: 40.0, phase: powerplay\nCONFIDENCE: MODERATE (n=300 balls)",
            "assistant": f"Attack the new ball with {ba} immediately. A strike rate of 145.0 in the powerplay suggests he can exploit the field restrictions. Keep the scoreboard ticking to stay ahead of the rate."
        })
    # 10: venue-based surface advice
    for _ in range(10):
        v = random.choice(VENUES)
        examples.append({
            "user": f"SITUATION: Toss won, deciding to bat or bowl at {v}.\nRESULTS: avg_first_innings_score: 175.0, surface_behavior_index: 0.8, avg_death_runs: 55.0\nCONFIDENCE: HIGH (n=45 matches)",
            "assistant": f"Bat first at {v}. The high surface behavior index of 0.8 favors batters early on, and an average first innings score of 175 is hard to chase here. Defend the total with scoreboard pressure."
        })
    # 10: pressure chase situations
    for _ in range(10):
        ba = random.choice(BATTERS)
        examples.append({
            "user": f"SITUATION: Chasing 40 in 18 balls, {ba} on strike.\nRESULTS: strike_rate: 165.0, dismissal_rate: 0.05, pressure_scenario: death_chase\nCONFIDENCE: LOW (n=45 balls)",
            "assistant": f"Trust {ba} to clear the ropes. His 165.0 strike rate in pressure scenarios is the weapon we need for this 13+ RPO chase. He must look for boundaries every second ball."
        })
    return examples[:count]

def format_message(system, user, assistant):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

# Generate data
sql_train = generate_sql_examples(100)
coach_train = generate_coach_examples(50)
train_raw = sql_train + coach_train
random.shuffle(train_raw)

sql_val = generate_sql_examples(20)
coach_val = generate_coach_examples(10)
val_raw = sql_val + coach_val
random.shuffle(val_raw)

def save_jsonl(data, path, is_sql):
    with open(path, "w") as f:
        for item in data:
            system = SCHEMA_PROMPT if "SQL:" in item["assistant"] else COACH_SYSTEM_PROMPT
            f.write(json.dumps(format_message(system, item["user"], item["assistant"])) + "\n")

save_jsonl(train_raw, TRAIN_PATH, True)
save_jsonl(val_raw, VAL_PATH, True)

print(f"Saved {len(train_raw)} examples to {TRAIN_PATH}")
print(f"Saved {len(val_raw)} examples to {VAL_PATH}")

# Validation logic
conn = sqlite3.connect(DB_PATH)
errors = 0
with open(TRAIN_PATH) as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        assistant = ex["messages"][-1]["content"]
        if "SQL:" in assistant:
            sql = assistant.split("SQL:")[-1].strip().rstrip(";") + ";"
            try:
                conn.execute(sql).fetchall()
            except Exception as e:
                print(f"Line {i} INVALID: {e}")
                print(f"SQL: {sql}")
                errors += 1
print(f"Validation complete: {errors} errors found.")
conn.close()
