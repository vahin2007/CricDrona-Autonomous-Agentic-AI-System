import sys
import json
import sqlite3
import argparse
from pathlib import Path

def load_eval_queries():
    # 20 queries with verifiable expectations
    return [
        {
            "query": "Who are the top 5 economy bowlers in death overs?",
            "situation": "death phase",
            "expected_table": "bowler_phase_stats",
            "expected_col": "economy",
        },
        {
            "query": "How has V Kohli performed in the powerplay?",
            "situation": "powerplay",
            "expected_table": "batter_phase_stats",
            "expected_col": "strike_rate",
        },
        {
            "query": "Show me JJ Bumrah stats vs MS Dhoni",
            "situation": "matchups",
            "expected_table": "matchups_1v1",
            "expected_col": "batter",
        },
        {
            "query": "What is the venue profile for Wankhede?",
            "situation": "venue",
            "expected_table": "venue_profiles",
            "expected_col": "venue",
        },
        {
            "query": "Who are the best death specialists?",
            "situation": "death phase",
            "expected_table": "bowler_phase_stats",
            "expected_col": "bowling_archetype",
        },
        {
            "query": "RG Sharma stats against spin bowling",
            "situation": "matchups",
            "expected_table": "batter_vs_bowling_style",
            "expected_col": "bowling_style",
        },
        {
            "query": "What is the recent form of Rashid Khan?",
            "situation": "recent form",
            "expected_table": "recent_form",
            "expected_col": "player",
        },
        {
            "query": "Who are the statistical clones for SP Narine?",
            "situation": "player clones",
            "expected_table": "player_clones",
            "expected_col": "clone",
        },
        {
            "query": "How does AB de Villiers play under pressure?",
            "situation": "pressure",
            "expected_table": "pressure_performance",
            "expected_col": "pressure_scenario",
        },
        {
            "query": "Best powerplay enforcers based on economy?",
            "situation": "powerplay",
            "expected_table": "bowler_phase_stats",
            "expected_col": "bowling_archetype",
        },
        {
            "query": "Top strike rates for middle overs?",
            "situation": "middle overs",
            "expected_table": "batter_phase_stats",
            "expected_col": "strike_rate",
        },
        {
            "query": "Eden Gardens average first innings score?",
            "situation": "venue",
            "expected_table": "venue_profiles",
            "expected_col": "avg_first_innings_score",
        },
        {
            "query": "Which players have an improving form trend?",
            "situation": "form",
            "expected_table": "recent_form",
            "expected_col": "form_trend",
        },
        {
            "query": "Stats for DA Warner vs pace",
            "situation": "matchups",
            "expected_table": "batter_vs_bowling_style",
            "expected_col": "bowling_style",
        },
        {
            "query": "How does M Chinnaswamy pitch behave?",
            "situation": "venue",
            "expected_table": "venue_profiles",
            "expected_col": "surface_behavior_index",
        },
        {
            "query": "Which bowlers take the most wickets in middle overs?",
            "situation": "middle overs",
            "expected_table": "bowler_phase_stats",
            "expected_col": "wickets",
        },
        {
            "query": "KL Rahul performance in pressure chases",
            "situation": "pressure",
            "expected_table": "pressure_performance",
            "expected_col": "pressure_scenario",
        },
        {
            "query": "Find clones for unknown player Arjun Tendulkar",
            "situation": "player clones",
            "expected_table": "player_clones",
            "expected_col": "clone",
        },
        {
            "query": "What is the dismissal rate for CH Gayle vs R Ashwin?",
            "situation": "matchups",
            "expected_table": "matchups_1v1",
            "expected_col": "dismissal_rate_wilson_lower",
        },
        {
            "query": "Stats for allrounders in the powerplay",
            "situation": "powerplay",
            "expected_table": "bowler_phase_stats",
            "expected_col": "bowling_archetype",
        },
    ]

def evaluate_model(model_name: str):
    sys.path.insert(0, "src")
    from agent import build_agent, run_query
    from langchain_ollama import OllamaLLM
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"Evaluating model: {model_name}")

    llm = OllamaLLM(model=model_name, base_url="http://localhost:11434", temperature=0.1, num_predict=600)
    agent = build_agent(llm)

    queries = load_eval_queries()
    results = []

    sql_execution_rate = 0
    schema_accuracy = 0
    result_accuracy = 0
    error_recovery_rate = 0

    for i, q in enumerate(queries):
        res = run_query(q["query"], agent, llm, q["situation"])
        
        executed_sql = res.get("sql", "") != "No SQL executed" and res.get("sql", "") != ""
        has_rows = len(res.get("rows", [])) > 0
        is_schema_accurate = False
        
        if executed_sql and "SELECT" in res.get("sql", "").upper():
            sql = res.get("sql").upper()
            if q["expected_table"].upper() in sql and q["expected_col"].upper() in sql:
                is_schema_accurate = True
        
        if executed_sql: sql_execution_rate += 1
        if is_schema_accurate: schema_accuracy += 1
        if has_rows: result_accuracy += 1
        
        results.append({
            "query": q["query"],
            "sql": res.get("sql", ""),
            "has_rows": has_rows,
            "is_schema_accurate": is_schema_accurate,
            "coach": res.get("coach", ""),
        })
        console.print(f"[{i+1}/{len(queries)}] {q['query'][:50]}... -> {'PASS' if has_rows else 'FAIL'}")

    metrics = {
        "model": model_name,
        "sql_execution_rate": sql_execution_rate / len(queries),
        "schema_accuracy": schema_accuracy / len(queries),
        "result_accuracy": result_accuracy / len(queries),
        "archetype_accuracy": 0.0, # Placeholder
        "phase_accuracy": 0.0, # Placeholder
        "error_recovery_rate": 0.0, # Placeholder
    }

    Path("db").mkdir(exist_ok=True)
    with open(f"db/eval_results_{model_name.replace(':', '_')}.json", "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics

def print_comparison():
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title="Model Evaluation Comparison")
    table.add_column("Metric", style="cyan")
    
    models = []
    for f in Path("db").glob("eval_results_*.json"):
        model_name = f.stem.replace("eval_results_", "").replace("_", ":")
        models.append(model_name)
        table.add_column(model_name, style="magenta")
        
    if not models:
        console.print("[red]No evaluation results found. Run evaluation first.[/red]")
        return

    data = {m: json.load(open(f"db/eval_results_{m.replace(':', '_')}.json"))["metrics"] for m in models}
    
    for metric in ["sql_execution_rate", "schema_accuracy", "result_accuracy"]:
        row = [metric]
        for m in models:
            row.append(f"{data[m][metric]:.2%}")
        table.add_row(*row)
        
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare all evaluated models")
    args = parser.parse_args()

    if args.compare:
        print_comparison()
    elif args.model:
        metrics = evaluate_model(args.model)
        print_comparison()
    else:
        print("Please specify a model with --model or use --compare")