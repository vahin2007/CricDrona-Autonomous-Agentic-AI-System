#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT DRONA — SQL ReAct Agent                                 ║
║  Layer 1: drona-cricket (via Ollama) → Text-based ReAct          ║
║  Layer 2: Same model, coach persona → tactical narration         ║
║                                                                  ║
║  Usage:                                                          ║
║    python src/agent.py --test      (run 5-query test suite)      ║
║    python src/agent.py             (interactive loop)            ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse, json, math, re, sqlite3, sys
from datetime import datetime
from pathlib import Path
from typing import Any
import sqlglot
from langchain_core.tools import tool as lc_tool
from langchain_ollama import OllamaLLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DB_PATH      = Path("db/cricket_drona.db")
MEMORY_PATH  = Path("db/learned_lessons.json")
MODEL_NAME   = "drona-cricket"
OLLAMA_URL   = "http://localhost:11434"
console      = Console()

# ─────────────────────────────────────────────────────────────────
# SCHEMA STRING — injected into every system prompt
# ─────────────────────────────────────────────────────────────────
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
7. CRITICAL: NEVER include 'TEXT', 'INT', or 'REAL' in your SELECT statements.
   BAD: SELECT batter TEXT, balls INT...
   GOOD: SELECT batter, balls_faced...
8. The column for dismissals rate is 'dismissal_rate_wilson_lower', NOT 'dismissal_rate'.
9. TABLE matchups_1v1 ONLY has 'runs_scored'. It DOES NOT have 'economy' or 'runs_conceded'.
10. TABLE bowler_phase_stats ONLY has 'economy'. It DOES NOT have 'runs_scored'.
11. NEVER put tool names like 'momentum_detector' or 'counterfactual_engine' inside a SELECT statement. Use ACTION instead.
12. ALWAYS use LIKE '%search_term%' for archetype matching to avoid case-sensitivity issues.
    Example: WHERE bowling_archetype LIKE '%death%'

!!! WARNING: 'matchups_1v1' table DOES NOT have 'venue'. To find venue stats, use 'venue_profiles'.

!!! WARNING: 'bowler_phase_stats' DOES NOT have 'batter'. It only tracks 'bowler'.
"""

COACH_PROMPT_TEMPLATE = """You are Drona — a gritty, no-nonsense T20 head coach.
The data analyst has already queried the database. The numbers below are verified facts.
Your ONLY job: give ONE sharp tactical recommendation based purely on these numbers.

RULES:
- Do NOT invent any statistic not shown below.
- Do NOT hedge — pick a side and defend it with the data.
- Be direct. Maximum 3 sentences.
- Start with the recommendation. Then cite the one key number. Then one risk if confidence is low.

MATCH SITUATION: {situation}

VERIFIED DATABASE RESULTS:
{results}

CONFIDENCE: {confidence_label} (n={sample_size} balls, Wilson CI lower: {wilson_ci:.3f})

PAST LESSONS FROM MEMORY:
{lessons}

Respond as Drona the coach:"""

REACT_INSTRUCTIONS = """
You have access to a database and these specific external functions:
  clone_lookup           — find clones for unknown players
  memory_read            — read past tactical lessons
  memory_write           — save a new lesson
  momentum_detector      — compute live match momentum
  team_weakness_profiler — find lineup weakness vs bowling styles
  counterfactual_engine  — compare two bowlers run differential

STRICT FORMAT — You MUST use one of these two formats exactly. NEVER use JSON. NEVER output curly braces {}.

Format 1: For Database Queries
THOUGHT: [what do I need?]
SQL: [exact SQL query]

Format 2: For External Functions
THOUGHT: [what do I need?]
ACTION: [function name]
INPUT: [input string]

After receiving an Observation, repeat. When you have enough data:
Final Answer: [your complete answer with stats and recommendation]

RULES:
- Always start with memory_read to check past lessons.
- If SQL returns an error, correct your SQL statement and try again.
- For lists or plural items (e.g. "top bowlers"), use LIMIT 5.
- SQL must use LIKE '%name%' for player names.
"""

# ─────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────

def wilson_lower(successes: float, trials: float, z: float = 1.96) -> float:
    if trials == 0:
        return 0.0
    p  = successes / trials
    z2 = z * z
    center = p + z2 / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    denom  = 1 + z2 / trials
    return max(0.0, min(1.0, round((center - margin) / denom, 4)))

def confidence_label(n: int) -> str:
    if n >= 60:
        return "HIGH CONFIDENCE"
    if n >= 25:
        return "MODERATE CONFIDENCE"
    return "LOW CONFIDENCE — small sample, treat as indicative only"

def compute_momentum(recent_balls: list[dict]) -> float:
    score = 0.0
    for ball in recent_balls[-12:]:
        score += -0.30 * int(ball.get("is_wicket", 0))
        score +=  0.15 * int(ball.get("is_boundary", 0))
        score += -0.05 * int(ball.get("is_dot", 0))
    return max(-1.0, min(1.0, round(score, 3)))

# ─────────────────────────────────────────────────────────────────
# EPISODIC MEMORY
# ─────────────────────────────────────────────────────────────────

def load_lessons(venue: str = "", phase: str = "") -> str:
    if not MEMORY_PATH.exists():
        return "No past lessons recorded yet."
    try:
        with open(MEMORY_PATH) as f:
            lessons: list[dict] = json.load(f)
    except Exception:
        return "Could not read memory file."
    relevant = []
    for lesson in lessons:
        v_match = venue.lower() in lesson.get("venue", "").lower() if venue else True
        p_match = phase.lower() in lesson.get("phase", "").lower() if phase else True
        if v_match or p_match:
            relevant.append(f"[{lesson.get('date','')}] {lesson.get('lesson','')}")
    if not relevant:
        return "No relevant past lessons for this context."
    return "\n".join(relevant[-5:])

def save_lesson(lesson_text: str, venue: str = "", phase: str = "") -> str:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lessons = []
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH) as f:
                lessons = json.load(f)
        except Exception:
            pass
    entry = {
        "date":   datetime.now().strftime("%Y-%m-%d"),
        "venue":  venue,
        "phase":  phase,
        "lesson": lesson_text.strip(),
    }
    lessons.append(entry)
    with open(MEMORY_PATH, "w") as f:
        json.dump(lessons, f, indent=2)
    return f"Lesson saved: {lesson_text[:80]}..."

# ─────────────────────────────────────────────────────────────────
# SQL GUARD
# ─────────────────────────────────────────────────────────────────

VALID_COLUMNS = {
    "matchups_1v1": {
        "batter","bowler","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
        "dismissal_rate_wilson_lower",
    },
    "bowler_phase_stats": {
        "bowler","phase","balls_bowled","runs_conceded","wickets",
        "economy","bowling_sr","average","dot_pct","boundary_concede_pct",
        "fours_conceded","sixes_conceded","bowling_archetype",
    },
    "batter_phase_stats": {
        "batter","phase","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
        "dismissal_rate_wilson_lower",
    },
    "venue_profiles": {
        "venue","matches_played","avg_first_innings_score",
        "avg_second_innings_score","avg_powerplay_runs","avg_middle_runs",
        "avg_death_runs","surface_behavior_index",
    },
    "batter_vs_bowling_style": {
        "batter","bowling_style","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
    },
    "recent_form": {
        "player","role","recent_avg_runs","recent_avg_sr",
        "recent_dismissal_rate","recent_economy","recent_wickets_per_match",
        "form_trend","sample_matches",
    },
    "player_clones": {"player","clone","similarity","clone_rank","role"},
    "pressure_performance": {
        "player","role","balls_faced","runs_scored","dismissals",
        "strike_rate","dot_pct","boundary_pct","dismissal_rate",
        "fours","sixes","pressure_scenario",
    },
}
VALID_TABLES = set(VALID_COLUMNS.keys())

def validate_sql(sql: str) -> tuple[bool, str]:
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as e:
        return False, f"SQL parse error: {e}"
    for table in parsed.find_all(sqlglot.exp.Table):
        tname = table.name.lower()
        if tname and tname not in VALID_TABLES:
            return False, f"Unknown table: '{tname}'. Valid: {sorted(VALID_TABLES)}"
    all_cols: set[str] = set()
    for table in parsed.find_all(sqlglot.exp.Table):
        all_cols.update(VALID_COLUMNS.get(table.name.lower(), set()))
    for col in parsed.find_all(sqlglot.exp.Column):
        cname = col.name.lower()
        if col.table:
            continue
        if all_cols and cname not in all_cols and cname != "*":
            return False, f"Unknown column: '{cname}'. Valid: {sorted(all_cols)}"
    return True, ""

# ─────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────

@lc_tool
def sql_executor(query: str) -> str:
    """Execute a SQL SELECT query against cricket_drona.db."""
    sql = query.strip()
    sql = re.sub(r"```sql\s*", "", sql, flags=re.I)
    sql = re.sub(r"```\s*", "", sql).strip().rstrip(";") + ";"
    if "counterfactual_engine" in sql.lower() or "momentum_detector" in sql.lower():
        return json.dumps({"error": "STOP! These are TOOLS, not SQL tables. Use ACTION: [tool_name] instead."})
    valid, err = validate_sql(sql)
    if not valid:
        try:
            from sql_memory import record_sql_error
            record_sql_error(sql, err, query_context=query)
        except Exception:
            pass
        return json.dumps({"error": err, "sql": sql, "rows": []})

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql).fetchall()]

        n = 0
        if rows:
            for key in ["balls_faced", "balls_bowled", "sample_matches"]:
                if key in rows[0]:
                    n = int(rows[0][key] or 0)
                    break

        wci = 0.0
        if rows and "dismissals" in rows[0] and n > 0:
            wci = wilson_lower(float(rows[0]["dismissals"] or 0), float(n))

        return json.dumps({
            "sql": sql, "rows": rows, "row_count": len(rows),
            "sample_size": n, "wilson_ci": wci,
            "confidence": confidence_label(n), "error": None,
        }, default=str)
    except sqlite3.Error as e:
        
        return json.dumps({"error": str(e), "sql": sql, "rows": []})

@lc_tool
def clone_lookup(player_name: str) -> str:
    """Find statistical clones for an unknown player."""
    sql = f"SELECT player, clone, similarity FROM player_clones WHERE player LIKE '%{player_name}%' LIMIT 3;"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
        return json.dumps({"clones": rows})
    except Exception as e:
        return json.dumps({"error": str(e), "clones": []})

@lc_tool
def memory_read(context: str) -> str:
    """Read past tactical lessons."""
    parts = context.lower().split()
    venue = " ".join([p for p in parts if len(p) > 4])
    phase = next((p for p in parts if p in ["powerplay", "middle", "death"]), "")
    return load_lessons(venue, phase)

@lc_tool
def memory_write(input_str: str) -> str:
    """Save a new lesson."""
    return save_lesson(input_str)

@lc_tool
def momentum_detector(balls_json: str) -> str:
    """Compute match momentum."""
    try:
        balls = json.loads(balls_json)
        score = compute_momentum(balls)
        return json.dumps({"momentum_score": score})
    except Exception as e:
        return json.dumps({"error": str(e)})

@lc_tool
def team_weakness_profiler(lineup_json: str) -> str:
    """Analyse collective batting weaknesses."""
    try:
        batters = json.loads(lineup_json)
        like_clauses = " OR ".join(["batter LIKE ?" for _ in batters])
        like_values  = [f"%{b}%" for b in batters]
        sql = f"SELECT bowling_style, AVG(dot_pct) as avg_dot FROM batter_vs_bowling_style WHERE ({like_clauses}) GROUP BY bowling_style;"
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, like_values).fetchall()]
        return json.dumps({"vulnerabilities": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})

@lc_tool
def counterfactual_engine(input_json: str) -> str:
    """Compare two bowler options."""
    try:
        data = json.loads(input_json)
        rec, alt, ph = data["recommended"], data["alternative"], data.get("phase", "death")
        sql = "SELECT bowler, economy FROM bowler_phase_stats WHERE (bowler LIKE ? OR bowler LIKE ?) AND phase = ?"
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = {r["bowler"]: dict(r) for r in conn.execute(sql, [f"%{rec}%", f"%{alt}%", ph]).fetchall()}
        return json.dumps({"comparison": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})

TOOL_MAP = {
    "sql_executor":          sql_executor.invoke,
    "clone_lookup":          clone_lookup.invoke,
    "memory_read":           memory_read.invoke,
    "memory_write":          memory_write.invoke,
    "momentum_detector":     momentum_detector.invoke,
    "team_weakness_profiler":team_weakness_profiler.invoke,
    "counterfactual_engine": counterfactual_engine.invoke,
}

# ─────────────────────────────────────────────────────────────────
# AGENT CORE (Text-based ReAct)
# ─────────────────────────────────────────────────────────────────

def build_agent(llm: OllamaLLM) -> OllamaLLM:
    return llm

def invoke_agent(llm: OllamaLLM, query: str, lessons: str) -> dict[str, Any]:
    try:
        from sql_memory import load_correction_rules
        correction_rules = load_correction_rules(top_n=8)
    except Exception:
        correction_rules = ""
    
    context = (
        SCHEMA_PROMPT
        + REACT_INSTRUCTIONS
        + (f"\n\nKNOWN MISTAKES TO AVOID:\n{correction_rules}\n" if correction_rules else "")
        + f"\n\nPAST LESSONS:\n{lessons}\n"
        + f"\nQuestion: {query}\n\n"
    )
    error_count=0
    conversation = context
    sql_result: dict = {}
    all_observations: list[str] = []

    for step in range(7):
        try:
            response = llm.invoke(conversation)
        except Exception as e:
            return {"output": f"LLM error: {e}", "sql_result": sql_result, "error": str(e)}

        if not isinstance(response, str): response = str(response)
        
        if len(all_observations) > 0 and "Final Answer:" not in response:
            drift_terms = ["kohli", "rohit", "sharma", "bumrah"]
            # Check if a drift term appeared that wasn't in the original user query
            if any(term in response.lower() and term not in query.lower() for term in drift_terms):
                return {
                    "output": "Tactical data retrieved. Transitioning to coach narration.", 
                    "sql_result": sql_result, 
                    "error": None
                }
        
        # 1. CHOP RAMBLING: Force the model to wait for an observation
        # --- THE TRIPLE-LOCK SANITIZER ---
        if "SQL:" in response:
            # Find the first SQL: and the first semicolon
            start = response.find("SQL:")
            end = response.find(";", start)
            if end != -1:
                response = response[:end + 1]
        elif "ACTION:" in response:
            # Find the first ACTION and its INPUT
            input_start = response.find("INPUT:")
            if input_start != -1:
                end = response.find("\n", input_start)
                response = response[:end] if end != -1 else response

        # 2. CURE AMNESIA: Save the LLM's thought to the conversation history!

        if response.strip() in conversation:
            conversation = context + f"\n[System Note: You are repeating yourself. Try a different SQL query.]\nQuestion: {query}\n"
            continue
        
        if len(all_observations) >= 3 and all("error" in obs.lower() for obs in all_observations[-3:]):
            conversation = context + f"\n[System: Resetting memory due to repeated errors. DO NOT use 'runs_conceded' in matchups_1v1. Use 'runs_scored'.]\nQuestion: {query}\n"
            all_observations = [] # Clear the observation tracker
            continue
        
        conversation += f"{response}\n"

        final_match  = re.search(r"Final Answer:\s*", response, re.IGNORECASE)

        tool_name = None
        tool_input = ""

        sql_match = re.search(r"SQL:\s*(SELECT.+?)$", response, re.IGNORECASE | re.MULTILINE)
        action_match = re.search(r"Action:\s*([a-zA-Z_]+)", response, re.IGNORECASE)

        first_match_start = float('inf')
        if sql_match: first_match_start = min(first_match_start, sql_match.start())
        if action_match: first_match_start = min(first_match_start, action_match.start())

        if final_match and (first_match_start == float('inf') or final_match.start() < first_match_start):
            output = response[final_match.end():].strip()
            return {"output": output, "sql_result": sql_result, "error": None}
        
        if sql_match and not (action_match and action_match.group(1).lower() == "sql_executor"):
            tool_name = "sql_executor"
            tool_input = sql_match.group(1).strip()
        elif action_match:
            tool_name = action_match.group(1).strip().lower()
            input_match  = re.search(r"(?:Action Input|Input):\s*(.+?)$", response, re.IGNORECASE | re.MULTILINE)
            tool_input = input_match.group(1).strip() if input_match else ""

        if not tool_name:
            conversation += "Observation: Please follow the format exactly. Use THOUGHT / SQL or THOUGHT / ACTION / INPUT.\n"
            continue

        if tool_name == "sql_executor":
            tool_input = re.sub(r"(?i)^```sql\s*", "", tool_input)
            tool_input = re.sub(r"^```\s*", "", tool_input)
            tool_input = re.sub(r"\s*```$", "", tool_input)
            if ";" in tool_input:
                tool_input = tool_input.split(";")[0] + ";"
            tool_input = tool_input.replace("\\", "").strip()

        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            clean_observation = f"Unknown tool '{tool_name}'. Available: {list(TOOL_MAP.keys())}"
            # Inside your loop in agent.py
            if "error" in clean_observation.lower():
                error_count += 1
                if error_count > 2:
                    return {"output": "I encountered multiple database errors. Please refine the query.", "sql_result": {}}
        else:
            try:
                raw_observation = tool_fn(tool_input)
                
                # 3. CURE JSON INFECTION: Translate JSON to plain English for the LLM
                if tool_name == "sql_executor":
                    try:
                        parsed = json.loads(raw_observation)
                        if parsed.get("error"):
                            clean_observation = f"SQL ERROR: {parsed['error']} - Correct your query and try again."
                        elif parsed.get("rows"):
                            sql_result = parsed  # Secretly save the JSON for the Coach!
                            clean_observation = f"SUCCESS: Found {len(parsed['rows'])} rows. Top results: {parsed['rows'][:3]}"
                        else:
                            clean_observation = "SUCCESS: Query executed but 0 rows were found."
                    except Exception:
                        clean_observation = str(raw_observation)
                else:
                    clean_observation = str(raw_observation)
                    
            except Exception as e:
                clean_observation = f"Tool execution error: {e}"
        # Inside your loop in agent.py
        if "error" in clean_observation.lower():
            error_count += 1
            if error_count > 2:
                return {"output": "I encountered multiple database errors. Please refine the query.", "sql_result": {}}
        all_observations.append(f"{tool_name}: {str(clean_observation)[:200]}")
        conversation += f"Observation: {clean_observation}\n"

    return {"output": "Reasoning timeout. " + "; ".join(all_observations[-2:]), "sql_result": sql_result, "error": "Timeout"}

def narrate_as_coach(llm: OllamaLLM, situation: str, sql_result: dict, lessons: str) -> str:
    rows = sql_result.get("rows", [])
    if not rows: return "Insufficient data to make a tactical recommendation."
    results_str = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows[:5])
    prompt = COACH_PROMPT_TEMPLATE.format(
        situation=situation, results=results_str, lessons=lessons,
        confidence_label=sql_result.get("confidence", "UNKNOWN"),
        sample_size=sql_result.get("sample_size", 0),
        wilson_ci=sql_result.get("wilson_ci", 0.0)
    )
    narrate_llm = OllamaLLM(model=llm.model, base_url=OLLAMA_URL, temperature=0.4, num_predict=200)
    result = narrate_llm.invoke(prompt)
    return result.strip() if isinstance(result, str) else str(result).strip()

def run_query(query: str, agent: OllamaLLM, llm: OllamaLLM, situation: str = "") -> dict[str, Any]:
    lessons    = load_lessons(venue=situation, phase="")
    agent_out  = invoke_agent(agent, query, lessons)
    sql_result = agent_out.get("sql_result", {})
    if agent_out.get("error") and not sql_result:
        return {"query": query, "sql": "", "rows": [], "coach": f"Agent error: {agent_out['error']}", "wilson_ci": 0.0, "sample_size": 0, "confidence": "ERROR", "lessons": lessons, "error": agent_out["error"]}
    coach = narrate_as_coach(llm, situation or query, sql_result, lessons)
    return {
        "query":       query,
        "sql":         sql_result.get("sql", "No SQL executed"),
        "rows":        sql_result.get("rows", []),
        "coach":       coach,
        "wilson_ci":   sql_result.get("wilson_ci", 0.0),
        "sample_size": sql_result.get("sample_size", 0),
        "confidence":  sql_result.get("confidence", "UNKNOWN"),
        "lessons":     lessons,
        "error":       sql_result.get("error"),
        "output":      agent_out.get("output", "No text generated."),
    }

def display_result(result: dict) -> None:
    console.rule("[bold cyan]DRONA TACTICAL OUTPUT[/bold cyan]")
    console.print(Panel(result["coach"], title="[bold green]Coach Advice[/bold green]", border_style="green"))
    ci_color = "green" if result["sample_size"] >= 60 else "yellow" if result["sample_size"] >= 25 else "red"
    console.print(f"[{ci_color}]Confidence: {result['confidence']} (n={result['sample_size']}, Wilson CI: {result['wilson_ci']:.3f})[/{ci_color}]")
    if result["sql"] and result["sql"] != "No SQL executed":
        console.print(Panel(result["sql"], title="[bold blue]SQL Provenance[/bold blue]", border_style="blue"))
    if result["rows"]:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        for col in result["rows"][0].keys(): tbl.add_column(str(col))
        for row in result["rows"][:8]: tbl.add_row(*[str(v) for v in row.values()])
        console.print(tbl)
    if result.get("error"): console.print(f"[red]Error: {result['error']}[/red]")

TEST_QUERIES = [
    {"query": "Who are the top economy bowlers in death overs?", "situation": "death overs economy"},
    {"query": "How does V Kohli bat in the powerplay?", "situation": "powerplay batter analysis"},
    {"query": "Give me JJ Bumrah head to head stats against RG Sharma", "situation": "matchup analysis"},
    {"query": "Which venue has the slowest surface behavior index?", "situation": "venue surface analysis"},
    {"query": "Who are the death specialists in the database?", "situation": "bowling archetype death overs"},
]

def run_test_suite(agent: OllamaLLM, llm: OllamaLLM) -> None:
    console.rule("[bold magenta]DRONA TEST SUITE — 5 QUERIES[/bold magenta]")
    passed = 0
    for i, test in enumerate(TEST_QUERIES, 1):
        console.print(f"\n[bold]Test {i}/5:[/bold] {test['query']}")
        result = run_query(query=test["query"], agent=agent, llm=llm, situation=test["situation"])
        if result["rows"]:
            console.print(f"[green]  PASS — {len(result['rows'])} rows returned[/green]")
            passed += 1
        else:
            console.print(f"[red]  FAIL — zero rows returned[/red]")
            if result.get("error"):
                console.print(f"  [red]Error: {result['error']}[/red]")
            console.print("\n[yellow]  --- AGENT INTERNAL MONOLOGUE ---[/yellow]")
            console.print(f"[dim]{result.get('output', 'No text generated.')}[/dim]")
            console.print("[yellow]  --------------------------------[/yellow]\n")
    console.rule()
    console.print(f"Result: {passed}/5 tests passed")

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Drona — SQL ReAct Agent")
    parser.add_argument("--test",  action="store_true", help="Run 5-query test suite")
    parser.add_argument("--model", default=MODEL_NAME,  help="Ollama model name")
    parser.add_argument("--query", type=str, default="", help="Single query")
    args = parser.parse_args()
    if not DB_PATH.exists(): return
    llm = OllamaLLM(model=args.model, base_url=OLLAMA_URL, temperature=0.1, num_predict=600,stop=["Observation:", "THOUGHT:", "SQL:"])
    agent = build_agent(llm)
    if args.test: run_test_suite(agent, llm); return
    if args.query:
        display_result(run_query(query=args.query, agent=agent, llm=llm))
        return
    console.print(Panel("Type your tactical question. 'quit' to exit | 'learn: <text>' to save lesson", title="Project Drona", border_style="cyan"))
    while True:
        try: query = console.input("\n[bold cyan]Drona >[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError): break
        if not query or query.lower() in ("quit", "exit", "q"): break
        if query.lower().startswith("learn:"):
            console.print(f"[yellow]{memory_write.invoke(query[6:].strip())}[/yellow]")
            continue
        situation = console.input("[dim]Match situation (optional): [/dim]").strip()
        display_result(run_query(query=query, agent=agent, llm=llm, situation=situation))

if __name__ == "__main__":
    main()
