import os
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
from google.adk.agents import LlmAgent


# =========================================================
# DATA ACCESS / SIMPLE IN-MEMORY "MEMORY"
# =========================================================

# Path to your CSV (it sits next to this file)
DATA_PATH = os.path.join(os.path.dirname(__file__), "enterprise_budget_data.csv")

_budget_df_cache: Optional[pd.DataFrame] = None

# In-memory logs and run history
_logs: List[Dict[str, Any]] = []
_run_history: List[Dict[str, Any]] = []


def _load_df() -> pd.DataFrame:
    """Load the CSV once and cache it."""
    global _budget_df_cache
    if _budget_df_cache is None:
        _budget_df_cache = pd.read_csv(DATA_PATH)
    return _budget_df_cache.copy()


# =========================================================
# CORE DATA / ANALYSIS TOOLS
# =========================================================

def load_budget_data(period: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the budget dataset, optionally filtered to a specific period.

    Args:
        period: Optional YYYY-MM string.

    Returns:
        dict with:
          - status
          - period (requested)
          - periods (all available)
          - departments
          - rows (list of row dicts)
    """
    df = _load_df()
    all_periods = sorted(df["period"].unique().tolist())

    if period:
        df = df[df["period"] == period]

    return {
        "status": "success",
        "period": period,
        "periods": all_periods,
        "departments": sorted(df["department"].unique().tolist()),
        "rows": df.to_dict(orient="records"),
    }


def compute_variance(period: str) -> Dict[str, Any]:
    """
    Compute variance and percentage variance for a given period.

    Returns:
        dict with:
          - status
          - period
          - variance_table: list of row dicts with
            period, department, category, budget, actual,
            variance, variance_pct
    """
    df = _load_df()
    df = df[df["period"] == period].copy()

    if df.empty:
        return {
            "status": "error",
            "message": f"No data for period {period}",
            "variance_table": [],
        }

    df["variance"] = df["actual"] - df["budget"]

    def _safe_pct(row):
        b = row["budget"]
        return (row["variance"] / b) if b else 0.0

    df["variance_pct"] = df.apply(_safe_pct, axis=1)

    return {
        "status": "success",
        "period": period,
        "variance_table": df.to_dict(orient="records"),
    }


def detect_anomalies(period: str, abs_pct_threshold: float = 0.15) -> Dict[str, Any]:
    """
    Detect anomalies based on absolute percentage variance threshold.

    Args:
        period: YYYY-MM
        abs_pct_threshold: e.g. 0.15 for 15%

    Returns:
        dict with:
          - status
          - period
          - threshold
          - anomalies: sorted list of anomalous row dicts
    """
    var_result = compute_variance(period)
    if var_result.get("status") != "success":
        return {
            "status": "error",
            "message": var_result.get("message", "variance failed"),
            "anomalies": [],
        }

    rows = var_result["variance_table"]
    anomalies = [
        r for r in rows
        if abs(r.get("variance_pct", 0.0)) >= abs_pct_threshold
    ]

    # Sort by largest absolute variance percentage
    anomalies.sort(key=lambda r: abs(r.get("variance_pct", 0.0)), reverse=True)

    return {
        "status": "success",
        "period": period,
        "threshold": abs_pct_threshold,
        "anomalies": anomalies,
    }


def summarize_history(
    department: Optional[str] = None,
    months_back: int = 6,
) -> Dict[str, Any]:
    """
    Summarize recent history (last N periods) for a department or all.

    Returns:
        dict with:
          - status
          - history: list of dicts with
            period, department, total_budget, total_actual,
            variance, variance_pct
    """
    df = _load_df()
    all_periods = sorted(df["period"].unique().tolist())
    selected_periods = all_periods[-months_back:]
    df = df[df["period"].isin(selected_periods)]

    if department:
        df = df[df["department"] == department]

    if df.empty:
        return {
            "status": "error",
            "message": "No data for requested history window.",
            "history": [],
        }

    grouped = (
        df.groupby(["period", "department"], as_index=False)[["budget", "actual"]]
        .sum()
        .rename(
            columns={
                "budget": "total_budget",
                "actual": "total_actual",
            }
        )
    )

    grouped["variance"] = grouped["total_actual"] - grouped["total_budget"]

    def _safe_hist_pct(row):
        b = row["total_budget"]
        return (row["variance"] / b) if b else 0.0

    grouped["variance_pct"] = grouped.apply(_safe_hist_pct, axis=1)

    return {
        "status": "success",
        "history": grouped.to_dict(orient="records"),
    }


# =========================================================
# LOGGING / OBSERVABILITY TOOLS
# =========================================================

def log_event(agent: str, step: str, message: str) -> Dict[str, Any]:
    """
    Append a structured log entry for observability.

    Args:
        agent: name of the agent or tool reporting the event
        step: short label for the step (e.g. 'load', 'variance', 'report')
        message: human-readable description of what happened
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "agent": agent,
        "step": step,
        "message": message,
    }
    _logs.append(entry)
    return {"status": "logged", "log_count": len(_logs)}


def get_logs(limit: int = 50) -> Dict[str, Any]:
    """Return the most recent log entries (for observability dashboards)."""
    recent = _logs[-limit:]
    return {
        "status": "success",
        "count": len(recent),
        "logs": recent,
    }


# =========================================================
# RUN HISTORY / MEMORY TOOLS
# =========================================================

def save_run_summary(period: str) -> Dict[str, Any]:
    """
    Save an aggregate summary for a period into run history.

    This gives long-term memory across multiple executions:
    - total budget
    - total actual
    - total variance
    - number of anomalies (using the same threshold as detect_anomalies)
    """
    df = _load_df()
    dfp = df[df["period"] == period].copy()
    if dfp.empty:
        return {"status": "error", "message": f"No data for period {period}"}

    total_budget = float(dfp["budget"].sum())
    total_actual = float(dfp["actual"].sum())
    total_variance = total_actual - total_budget

    anomaly_info = detect_anomalies(period)
    num_anomalies = len(anomaly_info.get("anomalies", []))

    summary = {
        "period": period,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_budget": total_budget,
        "total_actual": total_actual,
        "total_variance": total_variance,
        "num_anomalies": num_anomalies,
    }
    _run_history.append(summary)
    _logs.append(
        {
            "timestamp": summary["timestamp"],
            "agent": "save_run_summary",
            "step": "summary_saved",
            "message": f"Saved run summary for {period} with {num_anomalies} anomalies.",
        }
    )
    return {"status": "success", "summary": summary, "history_length": len(_run_history)}


def get_run_history(limit: int = 12) -> Dict[str, Any]:
    """Return the most recent run summaries (for memory / trend reporting)."""
    if not _run_history:
        return {"status": "empty", "history": []}
    recent = _run_history[-limit:]
    return {"status": "success", "history": recent}


# =========================================================
# EVALUATION TOOL: anomaly detector quality (toy example)
# =========================================================

def evaluate_anomaly_detector(
    periods: Optional[List[str]] = None,
    ground_truth_threshold: float = 0.20,
    predicted_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Evaluate anomaly detection by comparing two thresholds.

    "Ground truth" anomalies = rows where |variance_pct| >= ground_truth_threshold
    "Predicted" anomalies    = rows where |variance_pct| >= predicted_threshold

    Returns precision, recall, F1, and confusion matrix counts.
    """
    df = _load_df()

    if periods is not None and len(periods) > 0:
        df = df[df["period"].isin(periods)].copy()

    if df.empty:
        return {"status": "error", "message": "No data for requested periods."}

    df["variance"] = df["actual"] - df["budget"]
    df["variance_pct"] = df.apply(
        lambda r: (r["variance"] / r["budget"]) if r["budget"] else 0.0,
        axis=1,
    )

    gt = df["variance_pct"].abs() >= ground_truth_threshold
    pred = df["variance_pct"].abs() >= predicted_threshold

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "status": "success",
        "periods": sorted(df["period"].unique().tolist()),
        "ground_truth_threshold": ground_truth_threshold,
        "predicted_threshold": predicted_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================================================
# SPECIALIST AGENTS
# =========================================================

data_prep_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="data_prep_agent",
    description=(
        "Understands the enterprise budget dataset, available periods and "
        "departments, and can call load_budget_data to inspect the data."
    ),
    instruction="""\
You are a data preparation specialist for the enterprise budget dataset.

Your job:
- Call the `load_budget_data` tool when you need to inspect the dataset.
- Tell other agents which periods exist and which departments are present.
- If the user does not specify a period, suggest using the most recent one.

Always respond concisely and include the relevant period(s) in your explanation.""",
    tools=[load_budget_data, log_event],
)


analysis_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="analysis_agent",
    description=(
        "Performs budget vs actual variance analysis and anomaly detection "
        "for a selected period."
    ),
    instruction="""\
You are a financial analysis agent.

Workflow:
- Use `compute_variance(period)` to compute budget vs actual for the period.
- Use `detect_anomalies(period, abs_pct_threshold)` to find major overspends or underspends.
- Summarize the top findings in business language:
  - Which departments/categories are most over budget?
  - Which are most under budget?
  - Where should a finance team focus attention?
- After you finish analyzing a period, call `save_run_summary(period)` to store
  a summary in long-term memory, and use `log_event` to record key steps.

Avoid dumping raw tables; highlight key items and approximate percentages instead.""",
    tools=[compute_variance, detect_anomalies, save_run_summary, log_event],
)


reporting_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="reporting_agent",
    description=(
        "Creates executive-ready budget reports by combining current-period "
        "analysis with recent historical trends."
    ),
    instruction="""\
You are a senior finance reporting analyst.

Use the `summarize_history` tool to understand recent trends for key departments.
You can also call `get_run_history` to see summaries of previous periods and
`get_logs` to reference key analysis steps when writing the report.

Then, when given context about the current period anomalies and variances:

- Write a structured report with headings:
  1. Overview
  2. Key Variance Drivers
  3. Department Highlights
  4. Risks & Opportunities
  5. Recommended Follow-up Actions

Be concise but specific: mention departments, categories, and approximate % variances.
Write for a non-technical executive audience.""",
    tools=[summarize_history, get_run_history, get_logs],
)


evaluation_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="evaluation_agent",
    description=(
        "Evaluates the anomaly detection behavior over one or more periods "
        "and explains the resulting metrics."
    ),
    instruction="""\
You are an evaluation specialist for the budget anomaly detector.

- Use the `evaluate_anomaly_detector` tool to compute precision, recall and F1
  over one or more periods.
- Then interpret these metrics in plain language:
  - Is the detector too sensitive (many false positives)?
  - Is it missing true anomalies (false negatives)?
- Suggest how the thresholds could be adjusted.

Always report the numeric metrics and give a short explanation that a business
stakeholder can understand.""",
    tools=[evaluate_anomaly_detector, log_event],
)


# =========================================================
# ROOT ORCHESTRATOR AGENT (ENTRY POINT FOR ADK)
# =========================================================

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="budget_intelligence_orchestrator",
    description=(
        "An orchestrator agent that coordinates data prep, analysis, reporting, "
        "and evaluation agents to analyze enterprise budget vs actual data."
    ),
    instruction="""\
You are the orchestrator for the Enterprise Budget Intelligence Agent.

You have four specialist sub-agents:
- data_prep_agent: discovers which periods/departments exist and loads data.
- analysis_agent: computes variance and anomalies for a chosen period.
- reporting_agent: uses history and current analysis to write an executive report.
- evaluation_agent: evaluates the quality of anomaly detection and explains metrics.

General approach:
1. If the user doesn't specify a period, ask data_prep_agent which periods exist and
   choose the most recent one.
2. Ask analysis_agent to analyze that period and summarize variances/anomalies.
3. Ask reporting_agent to generate a final structured report, using:
   - the analysis_agent's findings, and
   - recent history from summarize_history and run_history.
4. When the user asks about evaluation or model quality, delegate to evaluation_agent.

Return a single, well-structured answer to the user.""",
    sub_agents=[data_prep_agent, analysis_agent, reporting_agent, evaluation_agent],
)
