"""
training_pipeline.py — End-to-end training orchestrator.

Phase 1 — Historical data (Excel 2023–2025)
  ├── Load & clean Excel files
  ├── Train XGBoost forecasters
  ├── Train Isolation Forest detectors
  └── Run PCMCI causal analysis

Phase 2 — SCADA simulation
  ├── Generate 50,000+ transitions
  └── Train Double DQN agent

Phase 3 — Backtest
  └── Compare Do-Nothing vs Random vs DQN

Run:  python backend/training_pipeline.py --phase all
"""

from __future__ import annotations
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database.database                       import init_db
from data_pipeline.excel_loader              import load_to_database
from data_pipeline.preprocessing             import clean_dataframe
from data_pipeline.feature_engineering       import add_all_features
from data_pipeline.data_validator            import validate
from forecasting.predictor_service           import PredictorService
from anomalies.isolation_forest_detector     import IsolationForestDetector
from causal.pcmci_engine                     import PCMCIEngine
from causal.dag_builder                      import build_dag_from_links
from rl.trainer                              import Trainer
from database.database                       import query, insert_many

import pandas as pd
import numpy as np


# ── Phase 1 ────────────────────────────────────────────────────────────────────

def phase1(data_dir: str = "data") -> dict:
    print("\n" + "="*60)
    print("PHASE 1 — Data Foundation & Core Intelligence")
    print("="*60)

    # 1a. Initialize DB and load Excel
    print("\n[1/5] Initializing database ...")
    init_db()

    print("\n[2/5] Loading Excel files ...")
    load_to_database(data_dir)

    # 1b. Load from DB and validate
    print("\n[3/5] Validating data ...")
    rows = query("SELECT * FROM historical_data ORDER BY timestamp")
    df   = pd.DataFrame(rows)
    report = validate(df)
    if not report.passed:
        for f in report.summary()["failed"]:
            print(f"  ⚠  {f['name']}: {f['detail']}")
    else:
        print(f"  ✓ All checks passed ({len(report.checks)} checks)")

    # 1c. Feature engineering
    df = clean_dataframe(df)
    df = add_all_features(df)
    print(f"  Features: {len(df.columns)} columns, {len(df):,} rows")

    # 1d. Train forecasters
    print("\n[4/5] Training XGBoost forecasters ...")
    ps      = PredictorService()
    ps_metrics = ps.train_all()

    # 1e. Train anomaly detectors
    print("\n  Training Isolation Forest detectors ...")
    ifd = IsolationForestDetector()
    ifd_metrics = ifd.train_all(df)

    # 1f. PCMCI causal analysis
    print("\n[5/5] Running PCMCI causal analysis ...")
    pcmci  = PCMCIEngine(max_lag=5)
    result = pcmci.run(df)
    print(f"  Found {len(result['edges'])} causal links")

    # Store causal links
    links = [{
        "source_var": e["source"], "target_var": e["target"],
        "lag": e["lag"], "strength": e["strength"], "p_value": e.get("p_value", 0),
    } for e in result["edges"][:50]]
    if links:
        insert_many("causal_links", links)

    print("\n✓ Phase 1 complete")
    return {"forecasting": ps_metrics, "anomaly": ifd_metrics,
            "causal_links": len(result["edges"])}


# ── Phase 2 ────────────────────────────────────────────────────────────────────

def phase2(episodes: int = 500) -> dict:
    print("\n" + "="*60)
    print("PHASE 2 — Double DQN Training")
    print("="*60)

    rows = query("SELECT * FROM historical_data ORDER BY timestamp")
    if rows:
        df   = pd.DataFrame(rows)
        df   = clean_dataframe(df)
        df   = add_all_features(df)
        data = df
    else:
        data = None
        print("  ⚠ No historical data — training on simulated transitions only")

    print(f"\n  Training {episodes} episodes ...")
    trainer = Trainer(data=data, episodes=episodes, episode_length=720)
    summary = trainer.train()

    print("\n✓ Phase 2 complete")
    print(f"  Best reward: {summary.get('best_reward', 0):.2f}")
    print(f"  Avg reward (last 50): {summary.get('avg_reward_last50', 0):.2f}")
    return summary


# ── Phase 3 ────────────────────────────────────────────────────────────────────

def phase3() -> dict:
    print("\n" + "="*60)
    print("PHASE 3 — Backtesting")
    print("="*60)

    rows = query("SELECT * FROM historical_data ORDER BY timestamp")
    if rows:
        df   = pd.DataFrame(rows)
        df   = clean_dataframe(df)
        df   = add_all_features(df)
        data = df
    else:
        data = None

    trainer = Trainer(data=data, episodes=1)   # load existing agent
    try:
        trainer.agent.load()
        print("  Loaded trained DQN agent")
    except Exception:
        print("  ⚠ No trained agent found — backtesting untrained agent")

    results = trainer.backtest(["do_nothing", "random", "dqn"])

    print("\n  Strategy comparison:")
    for strat, res in results.items():
        print(f"    {strat:<15} reward={res['total_reward']:>10.2f}")

    print("\n✓ Phase 3 complete")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCP Cogen Training Pipeline")
    parser.add_argument("--phase",    default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1(args.data_dir)
    if args.phase in ("2", "all"):
        phase2(args.episodes)
    if args.phase in ("3", "all"):
        phase3()