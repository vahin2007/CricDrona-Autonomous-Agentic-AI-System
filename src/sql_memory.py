# src/sql_memory.py
"""
Self-learning SQL correction system for Project Drona.
Tracks every SQL error, generalises patterns, injects corrections
into future prompts. The agent gets smarter with every mistake.
"""
from __future__ import annotations
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

ERROR_JOURNAL_PATH    = Path("db/sql_error_journal.json")
CORRECTION_RULES_PATH = Path("db/sql_correction_rules.json")

# Known valid schema — used to infer corrections automatically
VALID_COLUMNS = {
    "matchups_1v1": [
        "batter","bowler","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
        "dismissal_rate_wilson_lower",
    ],
    "bowler_phase_stats": [
        "bowler","phase","balls_bowled","runs_conceded","wickets",
        "economy","bowling_sr","average","dot_pct","boundary_concede_pct",
        "fours_conceded","sixes_conceded","bowling_archetype",
    ],
    "batter_phase_stats": [
        "batter","phase","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
        "dismissal_rate_wilson_lower",
    ],
    "venue_profiles": [
        "venue","matches_played","avg_first_innings_score",
        "avg_second_innings_score","avg_powerplay_runs","avg_middle_runs",
        "avg_death_runs","surface_behavior_index",
    ],
    "batter_vs_bowling_style": [
        "batter","bowling_style","balls_faced","runs_scored","dismissals",
        "strike_rate","average","dot_pct","boundary_pct","fours","sixes",
    ],
    "recent_form": [
        "player","role","recent_avg_runs","recent_avg_sr",
        "recent_dismissal_rate","recent_economy","recent_wickets_per_match",
        "form_trend","sample_matches",
    ],
    "player_clones": ["player","clone","similarity","clone_rank","role"],
    "pressure_performance": [
        "player","role","balls_faced","runs_scored","dismissals",
        "strike_rate","dot_pct","boundary_pct","dismissal_rate",
        "fours","sixes","pressure_scenario",
    ],
}

VALID_TABLES = set(VALID_COLUMNS.keys())

VALID_ENUM_VALUES = {
    "phase":             ["powerplay","middle","death"],
    "bowling_archetype": ["death_specialist","powerplay_enforcer",
                          "strike_bowler","economy_merchant","allrounder"],
    "bowling_style":     ["pace","spin","unknown"],
    "role":              ["batter","bowler"],
    "form_trend":        ["improving","declining","stable"],
}


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _find_closest_column(bad_col: str, table: str = "") -> str:
    """Find the closest valid column name using character overlap."""
    all_cols = []
    if table and table in VALID_COLUMNS:
        all_cols = VALID_COLUMNS[table]
    else:
        for cols in VALID_COLUMNS.values():
            all_cols.extend(cols)

    bad_lower = bad_col.lower()
    best, best_score = "", 0
    for col in all_cols:
        # Simple overlap score
        overlap = sum(1 for c in bad_lower if c in col.lower())
        score   = overlap / max(len(bad_lower), len(col))
        if score > best_score:
            best, best_score = col, score
    return best if best_score > 0.4 else "unknown"


def _extract_bad_column(error_msg: str) -> str:
    """Pull the bad column name out of a sqlglot error message."""
    match = re.search(r"Unknown column[:\s']+([a-zA-Z_]+)", error_msg, re.I)
    if match:
        return match.group(1)
    match = re.search(r"'([a-zA-Z_]+)' is not a valid column", error_msg, re.I)
    if match:
        return match.group(1)
    return ""


def _extract_bad_table(error_msg: str) -> str:
    """Pull the bad table name out of a sqlglot error message."""
    match = re.search(r"Unknown table[:\s']+([a-zA-Z_]+)", error_msg, re.I)
    if match:
        return match.group(1)
    return ""


def record_sql_error(bad_sql: str, error_msg: str, query_context: str = "") -> str:
    """
    Record a SQL error and generate a correction rule.
    Called by sql_executor whenever sqlglot rejects a query.
    Returns a human-readable correction hint to pass back to the agent.
    """
    # ── Layer A: Write to error journal ──────────────────────────
    journal = _load_json(ERROR_JOURNAL_PATH)
    existing = next((e for e in journal if e["bad_sql"] == bad_sql), None)
    if existing:
        existing["times_seen"] += 1
        existing["last_seen"]   = datetime.now().isoformat()
    else:
        journal.append({
            "bad_sql":       bad_sql,
            "error_msg":     error_msg,
            "query_context": query_context,
            "times_seen":    1,
            "first_seen":    datetime.now().isoformat(),
            "last_seen":     datetime.now().isoformat(),
        })
    _save_json(ERROR_JOURNAL_PATH, journal)

    # ── Layer B: Generate correction rule ─────────────────────────
    bad_col   = _extract_bad_column(error_msg)
    bad_table = _extract_bad_table(error_msg)
    hint      = ""

    if bad_col:
        correct_col = _find_closest_column(bad_col)
        rule_text   = (
            f"Do NOT use column '{bad_col}'. "
            f"The correct column name is '{correct_col}'."
        )
        hint = rule_text
        _upsert_rule(rule_text, bad_col, correct_col, bad_table or "any")

    elif bad_table:
        rule_text = (
            f"Do NOT use table '{bad_table}'. "
            f"Valid tables: {sorted(VALID_TABLES)}"
        )
        hint = rule_text
        _upsert_rule(rule_text, bad_table, "see valid tables", "schema")

    # Check for enum mistakes
    for col, valid_vals in VALID_ENUM_VALUES.items():
        if col in bad_sql.lower():
            # Find any quoted value that is NOT in the valid list
            quoted = re.findall(r"'([^']+)'", bad_sql)
            for q in quoted:
                if q not in valid_vals and len(q) > 2:
                    rule_text = (
                        f"The value '{q}' is not valid for column '{col}'. "
                        f"Valid values: {valid_vals}"
                    )
                    hint = rule_text
                    _upsert_rule(rule_text, q, str(valid_vals), col)

    return hint or f"SQL error: {error_msg}. Check column/table names against schema."


def _upsert_rule(rule_text: str, bad: str, correct: str, context: str) -> None:
    """Insert or update a correction rule."""
    rules = _load_json(CORRECTION_RULES_PATH)
    existing = next((r for r in rules if r["bad"] == bad), None)
    if existing:
        existing["times_seen"] += 1
        existing["rule"]        = rule_text
    else:
        rules.append({
            "rule":       rule_text,
            "bad":        bad,
            "correct":    correct,
            "context":    context,
            "times_seen": 1,
        })
    # Keep sorted by most frequent
    rules.sort(key=lambda x: x["times_seen"], reverse=True)
    _save_json(CORRECTION_RULES_PATH, rules)


def load_correction_rules(top_n: int = 10) -> str:
    """
    Load the top-N most frequent correction rules as a prompt string.
    Injected into the agent system prompt before every query.
    If no rules exist yet, returns empty string.
    """
    rules = _load_json(CORRECTION_RULES_PATH)
    if not rules:
        return ""
    top = rules[:top_n]
    lines = ["KNOWN SQL MISTAKES TO AVOID (learned from past errors):"]
    for r in top:
        lines.append(f"  - {r['rule']}  [seen {r['times_seen']} time(s)]")
    return "\n".join(lines)


def get_error_stats() -> dict:
    """Return summary statistics about SQL errors for the UI dashboard."""
    journal = _load_json(ERROR_JOURNAL_PATH)
    rules   = _load_json(CORRECTION_RULES_PATH)
    return {
        "total_errors":     len(journal),
        "unique_mistakes":  len(rules),
        "most_common":      rules[0]["bad"] if rules else "none",
        "total_corrections":sum(r["times_seen"] for r in rules),
    }