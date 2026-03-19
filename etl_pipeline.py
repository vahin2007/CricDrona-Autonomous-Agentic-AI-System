"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT DRONA — ETL PIPELINE                                    ║
║  Cricsheet Ashwin IPL format → cricket_drona.db (8 tables)      ║
║                                                                  ║
║  Input:  ~/drona/data/matches/  (*.csv + *_info.csv pairs)       ║
║          ~/drona/data/players.csv                                ║
║  Output: ~/drona/db/cricket_drona.db                             ║
║                                                                  ║
║  Usage:                                                          ║
║    python src/etl.py                                             ║
║    python src/etl.py --matches data/matches --db db/drona.db     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drona.etl")


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
class Config:
    MIN_H2H_BALLS:       int   = 6
    MIN_BATTER_BALLS:    int   = 30
    MIN_BOWLER_BALLS:    int   = 30
    MIN_CLONE_BALLS:     int   = 20
    MIN_PRESSURE_BALLS:  int   = 6
    RECENT_FORM_N:       int   = 10
    CLONE_TOP_K:         int   = 3
    N_ARCHETYPES:        int   = 5
    KMEANS_SEED:         int   = 42
    PRESSURE_MIN_TARGET: int   = 165
    PRESSURE_MIN_OVER:   int   = 16   # over_int >= 16 means overs 17-20


# ─────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────

def safe_div(num, den, fill: float = 0.0):
    """Vectorised division — returns fill wherever denominator is 0."""
    return np.where(den == 0, fill, num / den)


def wilson_lower(successes: pd.Series, trials: pd.Series, z: float = 1.96) -> pd.Series:
    """
    Wilson Score 95% CI lower bound on a binomial proportion.
    Used as the confidence score in the XAI layer.
    Returns 0.0 when trials == 0.
    """
    n  = trials.astype(float)
    p  = safe_div(successes.astype(float), n)
    z2 = z ** 2
    center = p + z2 / (2 * n)
    margin = z * np.sqrt(safe_div(p * (1 - p), n) + z2 / (4 * n * n))
    denom  = 1 + safe_div(pd.Series(np.ones(len(n))), pd.Series(n / z2))
    lower  = (center - margin) / (1 + z2 / np.where(n == 0, 1, n))
    return np.clip(pd.Series(lower), 0.0, 1.0).round(4)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — PARSE INFO FILES
# ─────────────────────────────────────────────────────────────────

def parse_info_file(info_path: Path) -> dict:
    """
    Parse the Cricsheet _info.csv key-value format.

    Format observed:
      version,2.1.0
      info,key,value
      info,team,Sunrisers Hyderabad
      info,winner,Sunrisers Hyderabad
      info,winner_runs,35
      info,player,TeamName,PlayerName
      info,registry,people,PlayerName,identifier

    Returns a flat dict with the keys we actually use.
    """
    meta = {
        "venue":           "",
        "city":            "",
        "season":          "",
        "date":            "",
        "event":           "",
        "match_number":    "",
        "toss_winner":     "",
        "toss_decision":   "",
        "winner":          "",
        "winner_runs":     None,
        "winner_wickets":  None,
        "player_of_match": "",
        "teams":           [],
    }

    try:
        with open(info_path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip().strip('"') for p in line.strip().split(",")]
                if len(parts) < 3:
                    continue
                if parts[0] != "info":
                    continue

                key = parts[1]
                val = parts[2] if len(parts) > 2 else ""

                if key == "venue":
                    # venue can contain commas — rejoin everything after key
                    meta["venue"] = ",".join(parts[2:]).strip().strip('"')
                elif key == "city":
                    meta["city"] = val
                elif key == "season":
                    meta["season"] = val
                elif key == "date":
                    meta["date"] = val
                elif key == "event":
                    meta["event"] = val
                elif key == "match_number":
                    meta["match_number"] = val
                elif key == "toss_winner":
                    meta["toss_winner"] = ",".join(parts[2:]).strip()
                elif key == "toss_decision":
                    meta["toss_decision"] = val
                elif key == "winner":
                    meta["winner"] = ",".join(parts[2:]).strip()
                elif key == "winner_runs":
                    try:
                        meta["winner_runs"] = int(val)
                    except ValueError:
                        pass
                elif key == "winner_wickets":
                    try:
                        meta["winner_wickets"] = int(val)
                    except ValueError:
                        pass
                elif key == "player_of_match":
                    meta["player_of_match"] = ",".join(parts[2:]).strip()
                elif key == "team":
                    meta["teams"].append(",".join(parts[2:]).strip())

    except Exception as e:
        log.debug("Could not parse info file %s: %s", info_path, e)

    return meta


# ─────────────────────────────────────────────────────────────────
# STEP 2 — LOAD ALL MATCHES
# ─────────────────────────────────────────────────────────────────

# Exact columns from your 1082591.csv header
BALL_COLS = {
    "match_id":              str,
    "season":                str,
    "start_date":            str,
    "venue":                 str,
    "innings":               "int8",
    "ball":                  str,
    "batting_team":          str,
    "bowling_team":          str,
    "striker":               str,
    "non_striker":           str,
    "bowler":                str,
    "runs_off_bat":          float,
    "extras":                float,
    "wides":                 float,
    "noballs":               float,
    "byes":                  float,
    "legbyes":               float,
    "penalty":               float,
    "wicket_type":           str,
    "player_dismissed":      str,
    "other_wicket_type":     str,
    "other_player_dismissed":str,
}


def load_all_matches(matches_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk the matches directory, load every ball-by-ball CSV and its
    corresponding _info.csv, return:
      - df_balls: all deliveries concatenated
      - df_meta:  one row per match with venue/winner/toss info
    """
    ball_files = sorted(
        p for p in matches_dir.glob("*.csv")
        if "_info" not in p.name
    )

    if not ball_files:
        raise FileNotFoundError(
            f"No ball-by-ball CSV files found in {matches_dir}\n"
            "Expected files like 1082591.csv alongside 1082591_info.csv"
        )

    log.info("Found %d match files in %s", len(ball_files), matches_dir)

    ball_chunks: list[pd.DataFrame] = []
    meta_rows:   list[dict]         = []

    for csv_path in tqdm(ball_files, desc="Loading matches", unit="match"):
        match_id = csv_path.stem  # "1082591"
        info_path = csv_path.parent / f"{match_id}_info.csv"

        # ── Load ball-by-ball ──────────────────────────────────────
        try:
            chunk = pd.read_csv(
                csv_path,
                dtype=str,          # read everything as str first
                na_values=[""],
                keep_default_na=False,
            )
            # Coerce numeric columns
            for col in ["runs_off_bat","extras","wides","noballs",
                        "byes","legbyes","penalty"]:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
                else:
                    chunk[col] = 0.0

            # Fill string columns
            for col in ["wicket_type","player_dismissed",
                        "other_wicket_type","other_player_dismissed"]:
                if col in chunk.columns:
                    chunk[col] = chunk[col].fillna("")
                else:
                    chunk[col] = ""

            ball_chunks.append(chunk)

        except Exception as e:
            log.warning("Skipping %s — read error: %s", csv_path.name, e)
            continue

        # ── Load info ──────────────────────────────────────────────
        if info_path.exists():
            meta = parse_info_file(info_path)
            meta["match_id"] = match_id
            meta_rows.append(meta)

    log.info("Concatenating %d match frames …", len(ball_chunks))
    df_balls = pd.concat(ball_chunks, ignore_index=True)
    df_meta  = pd.DataFrame(meta_rows)

    log.info(
        "Loaded %d deliveries │ %d matches │ seasons: %s",
        len(df_balls),
        df_balls["match_id"].nunique(),
        sorted(df_balls["season"].unique().tolist()),
    )
    return df_balls, df_meta


# ─────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all boolean flag columns and derived fields used downstream.
    All downstream builders receive this enriched DataFrame.
    """
    log.info("Engineering features …")

    # ── Boolean delivery flags ─────────────────────────────────────
    df["is_wide"]   = df["wides"] > 0
    df["is_noball"] = df["noballs"] > 0
    df["is_legal"]  = ~df["is_wide"]

    # Wicket fell on this ball (any kind)
    df["is_wicket"] = df["wicket_type"].str.strip().ne("")

    # Wicket charged to the BATTER (excludes run-outs of the non-striker)
    df["is_batter_out"] = df["is_wicket"] & ~df["wicket_type"].isin([
        "run out", "obstructing the field",
        "handled the ball", "timed out",
    ])

    df["is_boundary4"] = (df["runs_off_bat"] == 4) & df["is_legal"]
    df["is_boundary6"] = (df["runs_off_bat"] == 6) & df["is_legal"]

    # Dot ball: legal + zero bat runs + zero extras on this delivery
    df["is_dot"] = (
        df["is_legal"]
        & (df["runs_off_bat"] == 0)
        & (df["extras"] == 0)
    )

    # Total runs charged to the bowler on this delivery
    df["runs_conceded"] = (
        df["runs_off_bat"]
        + df["wides"]
        + df["noballs"]
        + df["byes"]
        + df["legbyes"]
    )

    # ── Phase assignment ───────────────────────────────────────────
    # ball format: "0.1" → over 0 (human: over 1)
    df["over_int"] = (
        df["ball"].str.split(".").str[0]
        .astype(int, errors="ignore")
    )
    # Coerce any non-numeric over_int to 0
    df["over_int"] = pd.to_numeric(df["over_int"], errors="coerce").fillna(0).astype(int)

    df["phase"] = pd.cut(
        df["over_int"],
        bins=[-1, 5, 14, 100],
        labels=["powerplay", "middle", "death"],
    ).astype(str)

    # ── Sort chronologically ───────────────────────────────────────
    df.sort_values(
        ["match_id", "innings", "over_int", "ball"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    log.info(
        "Features engineered │ %d legal deliveries │ %d wickets │ "
        "phase counts: %s",
        df["is_legal"].sum(),
        df["is_wicket"].sum(),
        df["phase"].value_counts().to_dict(),
    )
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 4 — BOWLING STYLE (heuristic — players.csv has no style col)
# ─────────────────────────────────────────────────────────────────

def attach_bowling_styles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer bowling style (pace / spin / unknown) from each bowler's
    statistical fingerprint. players.csv contains only identifier keys —
    no style column — so we derive it.

    Heuristic:
      Spinners in T20 tend to:
        - bowl mostly in middle overs (overs 7-15)  pp_pct < 0.22
        - generate higher dot% in middle  mid_dot_pct > 0.28
        - have lower economy variance (consistent, not pace-swing reliant)
        - lower overall economy in middle overs  < 7.8

    Returns df with 'bowling_style' column appended.
    """
    log.info("Inferring bowling styles (heuristic) …")

    legal = df[df["is_legal"]]

    feats = (
        legal.groupby("bowler")
        .agg(
            total_balls   = ("is_legal",  "sum"),
            pp_balls      = ("over_int",  lambda x: (x <= 5).sum()),
            mid_balls     = ("over_int",  lambda x: x.between(6, 14).sum()),
            mid_dots      = ("is_dot",
                             lambda x: x[
                                 legal.loc[x.index, "over_int"].between(6, 14)
                             ].sum()),
            total_runs    = ("runs_conceded", "sum"),
            mid_runs      = ("runs_conceded",
                             lambda x: x[
                                 legal.loc[x.index, "over_int"].between(6, 14)
                             ].sum()),
        )
        .reset_index()
    )

    feats["pp_pct"]       = safe_div(feats["pp_balls"],   feats["total_balls"])
    feats["mid_dot_pct"]  = safe_div(feats["mid_dots"],   feats["mid_balls"])
    feats["mid_economy"]  = safe_div(feats["mid_runs"] * 6, feats["mid_balls"])

    # Spin heuristic — conservative thresholds
    is_spin = (
        (feats["mid_dot_pct"]  > 0.28) &
        (feats["pp_pct"]       < 0.22) &
        (feats["mid_economy"]  < 7.8)  &
        (feats["total_balls"]  >= Config.MIN_BOWLER_BALLS)
    )
    feats["bowling_style"] = np.where(is_spin, "spin", "pace")
    feats.loc[feats["total_balls"] < Config.MIN_BOWLER_BALLS, "bowling_style"] = "unknown"

    style_map = feats.set_index("bowler")["bowling_style"].to_dict()
    df["bowling_style"] = df["bowler"].map(style_map).fillna("unknown")

    counts = feats["bowling_style"].value_counts()
    log.info(
        "  Style inference → pace: %d, spin: %d, unknown: %d",
        counts.get("pace", 0), counts.get("spin", 0), counts.get("unknown", 0),
    )
    return df


# ─────────────────────────────────────────────────────────────────
# TABLE BUILDERS
# ─────────────────────────────────────────────────────────────────

def build_matchups_1v1(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building matchups_1v1 …")
    legal = df[df["is_legal"]]

    g = legal.groupby(["striker", "bowler"])
    out = pd.DataFrame({
        "balls_faced":  g["is_legal"].sum(),
        "runs_scored":  g["runs_off_bat"].sum(),
        "dismissals":   g["is_batter_out"].sum(),
        "dot_balls":    g["is_dot"].sum(),
        "fours":        g["is_boundary4"].sum(),
        "sixes":        g["is_boundary6"].sum(),
    }).reset_index().rename(columns={"striker": "batter"})

    out = out[out["balls_faced"] >= Config.MIN_H2H_BALLS].copy()

    out["strike_rate"]  = safe_div(out["runs_scored"] * 100, out["balls_faced"]).round(2)
    out["average"]      = np.where(
        out["dismissals"] == 0,
        out["runs_scored"].astype(float),
        (out["runs_scored"] / out["dismissals"]).round(2),
    )
    out["dot_pct"]      = (safe_div(out["dot_balls"], out["balls_faced"]) * 100).round(2)
    out["boundary_pct"] = (safe_div(out["fours"] + out["sixes"], out["balls_faced"]) * 100).round(2)
    out["dismissal_rate_wilson_lower"] = wilson_lower(out["dismissals"], out["balls_faced"])

    log.info("  → %d matchup pairs", len(out))
    return out.sort_values(["batter", "balls_faced"], ascending=[True, False])


def build_bowler_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building bowler_phase_stats + archetypes …")

    legal = df[df["is_legal"]]

    # Economy uses ALL deliveries (wides cost runs)
    runs_all  = df.groupby(["bowler", "phase"])["runs_conceded"].sum()
    balls     = legal.groupby(["bowler", "phase"])["is_legal"].sum()
    wickets   = legal.groupby(["bowler", "phase"])["is_wicket"].sum()
    dots      = legal.groupby(["bowler", "phase"])["is_dot"].sum()
    fours_c   = legal.groupby(["bowler", "phase"])["is_boundary4"].sum()
    sixes_c   = legal.groupby(["bowler", "phase"])["is_boundary6"].sum()

    out = pd.concat(
        [runs_all, balls, wickets, dots, fours_c, sixes_c], axis=1
    )
    out.columns = [
        "runs_conceded", "balls_bowled", "wickets",
        "dot_balls", "fours_conceded", "sixes_conceded",
    ]
    out = out.reset_index()
    out = out[out["balls_bowled"] >= Config.MIN_BOWLER_BALLS].copy()

    out["economy"]              = safe_div(out["runs_conceded"] * 6, out["balls_bowled"]).round(2)
    out["bowling_sr"]           = safe_div(out["balls_bowled"], out["wickets"], fill=999.0).round(2)
    out["average"]              = safe_div(out["runs_conceded"], out["wickets"], fill=999.0).round(2)
    out["dot_pct"]              = (safe_div(out["dot_balls"], out["balls_bowled"]) * 100).round(2)
    out["boundary_concede_pct"] = (safe_div(out["fours_conceded"] + out["sixes_conceded"], out["balls_bowled"]) * 100).round(2)

    # ── K-Means bowling archetypes ─────────────────────────────────
    career = (
        out.groupby("bowler")
        .agg(
            c_economy    = ("economy",    "mean"),
            c_dot_pct    = ("dot_pct",    "mean"),
            c_bowl_sr    = ("bowling_sr", lambda x: x[x < 999].mean() if (x < 999).any() else 999.0),
            c_death_econ = ("economy",    lambda x: x[out.loc[x.index, "phase"] == "death"].mean()   if (out.loc[x.index, "phase"] == "death").any()    else 0.0),
            c_pp_econ    = ("economy",    lambda x: x[out.loc[x.index, "phase"] == "powerplay"].mean() if (out.loc[x.index, "phase"] == "powerplay").any() else 0.0),
        )
        .reset_index()
        .fillna(0)
    )

    FEAT = ["c_economy", "c_dot_pct", "c_bowl_sr", "c_death_econ", "c_pp_econ"]

    if len(career) >= Config.N_ARCHETYPES:
        scaler  = StandardScaler()
        X       = scaler.fit_transform(career[FEAT].values)
        km      = KMeans(
            n_clusters=Config.N_ARCHETYPES,
            random_state=Config.KMEANS_SEED,
            n_init=20, max_iter=500,
        )
        km.fit(X)
        career["cluster_id"] = km.labels_

        centroids = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=FEAT,
        )
        centroids["cluster_id"] = range(Config.N_ARCHETYPES)

        archetype_map: dict[int, str] = {}

        def claim(feature: str, asc: bool, label: str) -> None:
            avail = [c for c in centroids["cluster_id"] if c not in archetype_map]
            idx = (
                centroids[centroids["cluster_id"].isin(avail)]
                .sort_values(feature, ascending=asc)["cluster_id"].iloc[0]
            )
            archetype_map[idx] = label

        claim("c_death_econ", True,  "death_specialist")
        claim("c_pp_econ",    True,  "powerplay_enforcer")
        claim("c_bowl_sr",    True,  "strike_bowler")
        claim("c_economy",    True,  "economy_merchant")
        for cid in centroids["cluster_id"]:
            if cid not in archetype_map:
                archetype_map[cid] = "allrounder"

        career["bowling_archetype"] = career["cluster_id"].map(archetype_map)
        out = out.merge(career[["bowler", "bowling_archetype"]], on="bowler", how="left")
        out["bowling_archetype"] = out["bowling_archetype"].fillna("unknown")

        dist = career["bowling_archetype"].value_counts()
        log.info("  Archetypes → %s", dist.to_dict())
    else:
        log.warning("  Too few bowlers for K-Means (%d). All set to 'unknown'.", len(career))
        out["bowling_archetype"] = "unknown"

    log.info("  → %d bowler-phase rows", len(out))
    return out


def build_batter_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building batter_phase_stats …")
    legal = df[df["is_legal"]]

    g = legal.groupby(["striker", "phase"])
    out = pd.DataFrame({
        "balls_faced": g["is_legal"].sum(),
        "runs_scored": g["runs_off_bat"].sum(),
        "dismissals":  g["is_batter_out"].sum(),
        "dot_balls":   g["is_dot"].sum(),
        "fours":       g["is_boundary4"].sum(),
        "sixes":       g["is_boundary6"].sum(),
    }).reset_index().rename(columns={"striker": "batter"})

    out = out[out["balls_faced"] >= Config.MIN_BATTER_BALLS].copy()

    out["strike_rate"]  = safe_div(out["runs_scored"] * 100, out["balls_faced"]).round(2)
    out["average"]      = np.where(
        out["dismissals"] == 0,
        out["runs_scored"].astype(float),
        (out["runs_scored"] / out["dismissals"]).round(2),
    )
    out["dot_pct"]      = (safe_div(out["dot_balls"], out["balls_faced"]) * 100).round(2)
    out["boundary_pct"] = (safe_div(out["fours"] + out["sixes"], out["balls_faced"]) * 100).round(2)
    out["dismissal_rate_wilson_lower"] = wilson_lower(out["dismissals"], out["balls_faced"])

    log.info("  → %d batter-phase rows", len(out))
    return out


def build_venue_profiles(df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Venue-level scoring profiles + Surface Behavior Index.

    SBI = venue's median first-3-over economy / global median.
    SBI < 1.0 → slow/damp surface (favours bowlers).
    SBI > 1.0 → flat surface (favours batters).
    """
    log.info("Building venue_profiles …")

    # Per-innings totals
    inn_totals = (
        df.groupby(["match_id", "venue", "innings"])["runs_conceded"]
        .sum().reset_index(name="total_runs")
    )
    inn1 = inn_totals[inn_totals["innings"] == 1]
    inn2 = inn_totals[inn_totals["innings"] == 2]

    v1 = (
        inn1.groupby("venue")["total_runs"]
        .agg(avg_first_innings_score="mean", matches_played="count")
        .reset_index()
    )
    v2 = inn2.groupby("venue")["total_runs"].mean().rename("avg_second_innings_score")
    venue_stats = v1.join(v2, on="venue")

    # Phase-level average runs (first innings only — planning baseline)
    fi = df[df["innings"] == 1]
    phase_agg = (
        fi.groupby(["match_id", "venue", "phase"])["runs_conceded"]
        .sum().reset_index()
        .groupby(["venue", "phase"])["runs_conceded"]
        .mean()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    phase_agg.columns.name = None
    rename = {
        "powerplay": "avg_powerplay_runs",
        "middle":    "avg_middle_runs",
        "death":     "avg_death_runs",
    }
    for old, new in rename.items():
        if old not in phase_agg.columns:
            phase_agg[old] = 0.0
    phase_agg = phase_agg.rename(columns=rename)
    venue_stats = venue_stats.merge(
        phase_agg[["venue"] + list(rename.values())], on="venue", how="left"
    )

    # Surface Behavior Index
    early = fi[fi["over_int"] <= 2]
    early_by_match = (
        early.groupby(["match_id", "venue"])
        .agg(
            early_runs  = ("runs_conceded", "sum"),
            early_balls = ("is_legal", "sum"),
        )
        .reset_index()
    )
    early_by_match["early_balls"] = early_by_match["early_balls"].replace(0, 18)
    early_by_match["early_econ"]  = safe_div(
        early_by_match["early_runs"] * 6,
        early_by_match["early_balls"],
    )
    global_median = float(early_by_match["early_econ"].median())
    venue_early = (
        early_by_match.groupby("venue")["early_econ"]
        .median().reset_index(name="venue_early_econ")
    )
    venue_early["surface_behavior_index"] = safe_div(
        venue_early["venue_early_econ"],
        np.full(len(venue_early), global_median),
    ).round(3)

    venue_stats = venue_stats.merge(
        venue_early[["venue", "surface_behavior_index"]], on="venue", how="left"
    )
    venue_stats["surface_behavior_index"] = venue_stats["surface_behavior_index"].fillna(1.0)

    # Attach winner info from meta (most common winner per venue = "home advantage" proxy)
    if "winner" in df_meta.columns and "venue" in df_meta.columns:
        chase_wins = (
            df_meta[df_meta["toss_decision"] == "field"]
            .groupby("venue")
            .agg(
                chase_wins   = ("winner", "count"),
                total_tosses = ("winner", "count"),
            )
            .reset_index()
        )
        venue_stats = venue_stats.merge(chase_wins, on="venue", how="left")
        venue_stats["chase_wins"]   = venue_stats["chase_wins"].fillna(0).astype(int)
        venue_stats["total_tosses"] = venue_stats["total_tosses"].fillna(0).astype(int)

    for col in ["avg_first_innings_score", "avg_second_innings_score",
                "avg_powerplay_runs", "avg_middle_runs", "avg_death_runs"]:
        if col in venue_stats.columns:
            venue_stats[col] = venue_stats[col].round(1)

    log.info(
        "  → %d venues │ global SBI baseline economy: %.2f",
        len(venue_stats), global_median,
    )
    return venue_stats


def build_batter_vs_bowling_style(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building batter_vs_bowling_style …")
    legal = df[df["is_legal"]]

    g = legal.groupby(["striker", "bowling_style"])
    out = pd.DataFrame({
        "balls_faced": g["is_legal"].sum(),
        "runs_scored": g["runs_off_bat"].sum(),
        "dismissals":  g["is_batter_out"].sum(),
        "dot_balls":   g["is_dot"].sum(),
        "fours":       g["is_boundary4"].sum(),
        "sixes":       g["is_boundary6"].sum(),
    }).reset_index().rename(columns={"striker": "batter"})

    out = out[out["balls_faced"] >= Config.MIN_BATTER_BALLS].copy()
    out["strike_rate"]  = safe_div(out["runs_scored"] * 100, out["balls_faced"]).round(2)
    out["average"]      = np.where(
        out["dismissals"] == 0,
        out["runs_scored"].astype(float),
        (out["runs_scored"] / out["dismissals"]).round(2),
    )
    out["dot_pct"]      = (safe_div(out["dot_balls"], out["balls_faced"]) * 100).round(2)
    out["boundary_pct"] = (safe_div(out["fours"] + out["sixes"], out["balls_faced"]) * 100).round(2)

    log.info("  → %d rows", len(out))
    return out


def build_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building recent_form (last %d matches) …", Config.RECENT_FORM_N)

    df = df.sort_values("match_id")
    legal = df[df["is_legal"]]
    n = Config.RECENT_FORM_N

    # ── Batter ─────────────────────────────────────────────────────
    b_match = (
        legal.groupby(["striker", "match_id"])
        .agg(
            innings_runs  = ("runs_off_bat", "sum"),
            balls_faced   = ("is_legal",     "sum"),
            outs          = ("is_batter_out","sum"),
        )
        .reset_index()
    )
    b_match["innings_sr"] = safe_div(b_match["innings_runs"] * 100, b_match["balls_faced"])

    def bat_trend(grp: pd.DataFrame) -> str:
        if len(grp) < 4:
            return "stable"
        mid = len(grp) // 2
        return (
            "improving" if grp["innings_runs"].iloc[mid:].mean() > grp["innings_runs"].iloc[:mid].mean() * 1.15
            else "declining" if grp["innings_runs"].iloc[mid:].mean() < grp["innings_runs"].iloc[:mid].mean() * 0.85
            else "stable"
        )

    batter_rows = []
    for player, grp in b_match.groupby("striker"):
        grp = grp.tail(n)
        batter_rows.append({
            "player":                  player,
            "role":                    "batter",
            "recent_avg_runs":         round(float(grp["innings_runs"].mean()), 2),
            "recent_avg_sr":           round(float(grp["innings_sr"].mean()), 2),
            "recent_dismissal_rate":   round(float(grp["outs"].mean()), 3),
            "recent_economy":          None,
            "recent_wickets_per_match":None,
            "form_trend":              bat_trend(grp),
            "sample_matches":          len(grp),
        })

    # ── Bowler ─────────────────────────────────────────────────────
    w_match = (
        df.groupby(["bowler", "match_id"])
        .agg(
            runs_conceded = ("runs_conceded", "sum"),
            balls_bowled  = ("is_legal",      "sum"),
            wickets       = ("is_wicket",     "sum"),
        )
        .reset_index()
    )
    w_match["economy"] = safe_div(w_match["runs_conceded"] * 6, w_match["balls_bowled"])

    def bowl_trend(grp: pd.DataFrame) -> str:
        if len(grp) < 4:
            return "stable"
        mid = len(grp) // 2
        return (
            "improving" if grp["economy"].iloc[mid:].mean() < grp["economy"].iloc[:mid].mean() * 0.90
            else "declining" if grp["economy"].iloc[mid:].mean() > grp["economy"].iloc[:mid].mean() * 1.10
            else "stable"
        )

    bowler_rows = []
    for player, grp in w_match.groupby("bowler"):
        grp = grp.tail(n)
        bowler_rows.append({
            "player":                  player,
            "role":                    "bowler",
            "recent_avg_runs":         None,
            "recent_avg_sr":           None,
            "recent_dismissal_rate":   None,
            "recent_economy":          round(float(grp["economy"].mean()), 2),
            "recent_wickets_per_match":round(float(grp["wickets"].mean()), 3),
            "form_trend":              bowl_trend(grp),
            "sample_matches":          len(grp),
        })

    out = pd.DataFrame(batter_rows + bowler_rows)
    log.info("  → %d player-role rows", len(out))
    return out


def build_pressure_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-pressure situations: second innings, overs 17–20,
    when the first innings total was >= PRESSURE_MIN_TARGET.
    """
    log.info("Building pressure_performance …")

    fi_scores = (
        df[df["innings"] == 1]
        .groupby("match_id")["runs_conceded"]
        .sum().reset_index(name="fi_score")
    )
    pressure_matches = fi_scores[
        fi_scores["fi_score"] >= Config.PRESSURE_MIN_TARGET
    ]["match_id"].tolist()

    p_df = df[
        (df["innings"] == 2)
        & (df["over_int"] >= Config.PRESSURE_MIN_OVER)
        & (df["match_id"].isin(pressure_matches))
        & df["is_legal"]
    ].copy()

    if len(p_df) == 0:
        log.warning("  No pressure deliveries found — lower PRESSURE_MIN_TARGET?")
        return pd.DataFrame()

    bg = p_df.groupby("striker")
    bp = pd.DataFrame({
        "balls_faced": bg["is_legal"].sum(),
        "runs_scored": bg["runs_off_bat"].sum(),
        "dismissals":  bg["is_batter_out"].sum(),
        "fours":       bg["is_boundary4"].sum(),
        "sixes":       bg["is_boundary6"].sum(),
        "dot_balls":   bg["is_dot"].sum(),
    }).reset_index().rename(columns={"striker": "player"})
    bp["role"] = "batter"

    wg = p_df.groupby("bowler")
    wp = pd.DataFrame({
        "balls_faced": wg["is_legal"].sum(),
        "runs_scored": wg["runs_conceded"].sum(),
        "dismissals":  wg["is_wicket"].sum(),
        "fours":       wg["is_boundary4"].sum(),
        "sixes":       wg["is_boundary6"].sum(),
        "dot_balls":   wg["is_dot"].sum(),
    }).reset_index().rename(columns={"bowler": "player"})
    wp["role"] = "bowler"

    out = pd.concat([bp, wp], ignore_index=True)
    out = out[out["balls_faced"] >= Config.MIN_PRESSURE_BALLS].copy()

    out["strike_rate"]    = safe_div(out["runs_scored"] * 100, out["balls_faced"]).round(2)
    out["dot_pct"]        = (safe_div(out["dot_balls"], out["balls_faced"]) * 100).round(2)
    out["boundary_pct"]   = (safe_div(out["fours"] + out["sixes"], out["balls_faced"]) * 100).round(2)
    out["dismissal_rate"] = safe_div(out["dismissals"], out["balls_faced"]).round(4)
    out["pressure_scenario"] = f"inn2_over{Config.PRESSURE_MIN_OVER}+_target>={Config.PRESSURE_MIN_TARGET}"

    log.info(
        "  → %d player-role rows from %d pressure deliveries",
        len(out), len(p_df),
    )
    return out


def build_player_clones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity OOD fallback table.
    When an uncapped / unknown player is queried, the agent
    finds their closest statistical twin in the DB and queries that.
    """
    log.info("Building player_clones …")
    legal = df[df["is_legal"]]

    # ── Batter feature vectors ─────────────────────────────────────
    b_g = (
        legal.groupby("striker")
        .agg(
            balls        = ("is_legal",     "sum"),
            runs         = ("runs_off_bat",  "sum"),
            outs         = ("is_batter_out","sum"),
            dots         = ("is_dot",       "sum"),
            fours        = ("is_boundary4", "sum"),
            sixes        = ("is_boundary6", "sum"),
        )
        .reset_index()
    )
    b_g = b_g[b_g["balls"] >= Config.MIN_CLONE_BALLS].copy()
    b_g["sr"]           = safe_div(b_g["runs"] * 100, b_g["balls"])
    b_g["average"]      = np.where(b_g["outs"] == 0, b_g["runs"].astype(float),
                                   b_g["runs"] / b_g["outs"])
    b_g["dot_pct"]      = safe_div(b_g["dots"], b_g["balls"]) * 100
    b_g["boundary_pct"] = safe_div(b_g["fours"] + b_g["sixes"], b_g["balls"]) * 100

    b_phase = (
        legal.groupby(["striker", "phase"])
        .agg(p_balls=("is_legal", "sum"), p_runs=("runs_off_bat", "sum"))
        .reset_index()
    )
    b_phase["phase_sr"] = safe_div(b_phase["p_runs"] * 100, b_phase["p_balls"])
    b_pw = (
        b_phase.pivot_table(
            index="striker", columns="phase", values="phase_sr", fill_value=0
        ).reset_index()
    )
    for c in ["powerplay", "middle", "death"]:
        if c not in b_pw.columns:
            b_pw[c] = 0.0
    b_pw.rename(columns={"powerplay":"pp_sr","middle":"mid_sr","death":"death_sr"}, inplace=True)

    BFEAT = ["sr","average","dot_pct","boundary_pct","pp_sr","mid_sr","death_sr"]
    batter_feat = (
        b_g[["striker","sr","average","dot_pct","boundary_pct"]]
        .merge(b_pw[["striker","pp_sr","mid_sr","death_sr"]], on="striker", how="left")
        .fillna(0)
    )

    # ── Bowler feature vectors ─────────────────────────────────────
    w_g = (
        df.groupby("bowler")
        .agg(
            balls = ("is_legal",      "sum"),
            runs  = ("runs_conceded", "sum"),
            wkts  = ("is_wicket",     "sum"),
            dots  = ("is_dot",        "sum"),
        )
        .reset_index()
    )
    w_g = w_g[w_g["balls"] >= Config.MIN_CLONE_BALLS].copy()
    w_g["economy"]     = safe_div(w_g["runs"] * 6, w_g["balls"])
    w_g["dot_pct"]     = safe_div(w_g["dots"], w_g["balls"]) * 100
    w_g["wicket_rate"] = safe_div(w_g["wkts"], w_g["balls"]) * 100

    w_phase = (
        df.groupby(["bowler", "phase"])
        .agg(p_balls=("is_legal","sum"), p_runs=("runs_conceded","sum"))
        .reset_index()
    )
    w_phase["phase_econ"] = safe_div(w_phase["p_runs"] * 6, w_phase["p_balls"])
    w_pw = (
        w_phase.pivot_table(
            index="bowler", columns="phase", values="phase_econ", fill_value=0
        ).reset_index()
    )
    for c in ["powerplay", "middle", "death"]:
        if c not in w_pw.columns:
            w_pw[c] = 0.0
    w_pw.rename(columns={"powerplay":"pp_econ","middle":"mid_econ","death":"death_econ"}, inplace=True)

    WFEAT = ["economy","dot_pct","wicket_rate","pp_econ","mid_econ","death_econ"]
    bowler_feat = (
        w_g[["bowler","economy","dot_pct","wicket_rate"]]
        .merge(w_pw[["bowler","pp_econ","mid_econ","death_econ"]], on="bowler", how="left")
        .fillna(0)
    )

    # ── Cosine similarity computation ──────────────────────────────
    def compute_clones(
        features_df: pd.DataFrame,
        id_col: str,
        feat_cols: list,
        role: str,
    ) -> pd.DataFrame:
        if len(features_df) < 2:
            return pd.DataFrame()
        scaler  = StandardScaler()
        X       = scaler.fit_transform(features_df[feat_cols].values)
        sim_mat = cosine_similarity(X)
        np.fill_diagonal(sim_mat, -1.0)
        players = features_df[id_col].tolist()
        rows = []
        for i, player in enumerate(players):
            top_idx = np.argsort(sim_mat[i])[-Config.CLONE_TOP_K:][::-1]
            for rank, j in enumerate(top_idx, start=1):
                score = float(sim_mat[i][j])
                if score > 0:
                    rows.append({
                        "player":     player,
                        "clone":      players[j],
                        "similarity": round(score, 4),
                        "clone_rank": rank,
                        "role":       role,
                    })
        return pd.DataFrame(rows)

    bc = compute_clones(batter_feat, "striker", BFEAT, "batter")
    wc = compute_clones(bowler_feat, "bowler",  WFEAT, "bowler")
    out = pd.concat([bc, wc], ignore_index=True)
    log.info("  → %d clone pairs (top-%d per player)", len(out), Config.CLONE_TOP_K)
    return out


# ─────────────────────────────────────────────────────────────────
# SQLITE WRITER
# ─────────────────────────────────────────────────────────────────

INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_mu_batter      ON matchups_1v1(batter);",
    "CREATE INDEX IF NOT EXISTS idx_mu_bowler      ON matchups_1v1(bowler);",
    "CREATE INDEX IF NOT EXISTS idx_mu_bb          ON matchups_1v1(batter,bowler);",
    "CREATE INDEX IF NOT EXISTS idx_bps_bowler     ON bowler_phase_stats(bowler);",
    "CREATE INDEX IF NOT EXISTS idx_bps_phase      ON bowler_phase_stats(phase);",
    "CREATE INDEX IF NOT EXISTS idx_bps_archetype  ON bowler_phase_stats(bowling_archetype);",
    "CREATE INDEX IF NOT EXISTS idx_batps_batter   ON batter_phase_stats(batter);",
    "CREATE INDEX IF NOT EXISTS idx_batps_phase    ON batter_phase_stats(phase);",
    "CREATE INDEX IF NOT EXISTS idx_venue          ON venue_profiles(venue);",
    "CREATE INDEX IF NOT EXISTS idx_bvbs_batter    ON batter_vs_bowling_style(batter);",
    "CREATE INDEX IF NOT EXISTS idx_bvbs_style     ON batter_vs_bowling_style(bowling_style);",
    "CREATE INDEX IF NOT EXISTS idx_form_player    ON recent_form(player);",
    "CREATE INDEX IF NOT EXISTS idx_clone_player   ON player_clones(player);",
    "CREATE INDEX IF NOT EXISTS idx_clone_role     ON player_clones(role);",
    "CREATE INDEX IF NOT EXISTS idx_pressure_p     ON pressure_performance(player);",
]


def write_db(tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing to %s …", db_path)

    with sqlite3.connect(db_path) as conn:
        for name, tbl in tables.items():
            if tbl is None or tbl.empty:
                log.warning("  SKIP %-30s (empty)", name)
                continue
            tbl.to_sql(name, conn, if_exists="replace", index=False)
            log.info("  ✔  %-30s  %6d rows  %d cols", name, len(tbl), tbl.shape[1])

        cur = conn.cursor()
        for ddl in INDEX_DDL:
            try:
                cur.execute(ddl)
            except sqlite3.OperationalError as e:
                log.debug("Index skipped: %s", e)
        conn.commit()

    # Final validation
    log.info("─" * 60)
    log.info("DB VALIDATION:")
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (t,) in cur.fetchall():
            cur.execute(f"SELECT COUNT(*) FROM [{t}];")
            count = cur.fetchone()[0]
            status = "OK" if count > 0 else "EMPTY — CHECK THIS"
            log.info("  %-30s  %6d rows  [%s]", t, count, status)

    size_mb = db_path.stat().st_size / 1e6
    log.info("DB size: %.1f MB", size_mb)


# ─────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

def run(matches_dir: Path, db_path: Path) -> None:
    log.info("═" * 60)
    log.info("  PROJECT DRONA — ETL START")
    log.info("  Matches dir : %s", matches_dir)
    log.info("  Output DB   : %s", db_path)
    log.info("═" * 60)

    # Step 1: Load all match files
    df_balls, df_meta = load_all_matches(matches_dir)

    # Step 2: Feature engineering
    df = engineer_features(df_balls)

    # Step 3: Attach bowling styles
    df = attach_bowling_styles(df)

    # Step 4: Build all 8 tables
    tables = {
        "matchups_1v1":            build_matchups_1v1(df),
        "bowler_phase_stats":      build_bowler_phase_stats(df),
        "batter_phase_stats":      build_batter_phase_stats(df),
        "venue_profiles":          build_venue_profiles(df, df_meta),
        "batter_vs_bowling_style": build_batter_vs_bowling_style(df),
        "recent_form":             build_recent_form(df),
        "player_clones":           build_player_clones(df),
        "pressure_performance":    build_pressure_performance(df),
    }

    # Step 5: Write to SQLite
    write_db(tables, db_path)

    log.info("═" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project Drona ETL")
    p.add_argument(
        "--matches", type=Path,
        default=Path("data/matches"),
        help="Directory containing *.csv and *_info.csv files",
    )
    p.add_argument(
        "--db", type=Path,
        default=Path("db/cricket_drona.db"),
        help="Output SQLite database path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(matches_dir=args.matches, db_path=args.db)
