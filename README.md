![Title-Thumbnail](images/Title-Thumbnail-main.png)

# Enterprise Budget Intelligence Agent

A Multi-Agent System for Automated Financial Variance & Anomaly Analysis Using Google ADK + Gemini

## 🧠 Overview

The Enterprise Budget Intelligence Agent is a multi-agent financial analysis system built using the Google AI Agent Development Kit (ADK).
It automatically:
- Loads enterprise budget data
- Computes budget vs actual variances
- Detects anomalies (overspend/underspend)
- Summarizes multi-month trends
- Logs all steps for observability
- Stores period summaries in memory
- Evaluates anomaly detection performance
- Produces an executive-ready financial report
This project transforms the monthly financial review into a fully automated, AI-driven workflow.

## 🚨 Problem Statement

Enterprises handle large amounts of budget vs. actual data every month.
Analyzing department-wise performance requires:

- manually pulling spreadsheets
- identifying anomalies
- summarizing variance drivers
- generating executive reports
This is slow, labor-intensive, error-prone, and difficult at scale.

### The challenge:

“How can we automate financial analysis reliably and generate high-quality insights consistently?”

✅ Solution — Multi-Agent Financial Intelligence System

This project uses multiple LLM-powered agents—each specializing in one part of the workflow—to deliver reliable, explainable, and scalable financial reporting.

✔ Multi-Agent Architecture

- root_agent – Orchestrates the workflow
- data_prep_agent – Loads dataset & identifies periods
- analysis_agent – Computes variances & detects anomalies
- reporting_agent – Generates executive-ready report
- evaluation_agent – Measures anomaly detection model quality

✔ Custom Tools

- compute_variance
- detect_anomalies
- summarize_history
- save_run_summary (memory)
- log_event (logging)
- evaluate_anomaly_detector (agent evaluation)

✔ Memory

- Stores summaries for each analyzed period
- Used for trend reporting in subsequent runs

✔ Observability

- Full structured logs available via get_logs()
  
✔ Evaluation

- Precision / Recall / F1 for anomaly detection

## 🏗️ Architecture Diagram

![architecture](images/architecture.png)

.
## 🧭 Workflow Overview

![workflow](images/workflow.png)

The Enterprise Budget Intelligence workflow is shown in the diagram and works as follows:

1. User → Root Multi-agent Orchestrator

- A finance user starts in the ADK Web UI or CLI and asks a question such as:
“Give me an executive budget report for December 2024.”
- The request is sent to the Root Multi-agent Orchestrator, an LLM agent powered by Gemini (Google AI).

2. Orchestrator → Data Prep Agent

- The orchestrator first delegates to the Data Prep Agent.
- This agent uses the load_budget_data tool to:
  - Load the CSV data (enterprise_budget_data.csv)
  - Discover which periods and departments exist
- It may log this step using log_event and sends the cleaned context back to the orchestrator.

3. Orchestrator → Analysis Agent

- Next, the orchestrator calls the Analysis Agent with the selected period (for ex, 2024-12).
- The Analysis Agent uses:
  - compute_variance to calculate budget vs actual for each department/category
  - detect_anomalies to find major over- and under-spends
- It then calls save_run_summary, which writes a run summary into Memory, and logs key steps via log_event.

4. Memory Layer (Logs + Local Data + Run History)

- Local Data: original CSV, accessed via _load_df() and tools
- Run History: period summaries stored by save_run_summary
- Logs: structured log entries written by log_event
- This layer is the shared “state” that agents can read from to keep context over time.

5. Orchestrator → Reporting Agent

- With analysis complete and memory updated, the orchestrator delegates to the Reporting Agent.
- The Reporting Agent uses:
  - summarize_history to see recent trends
  - get_run_history to compare current period vs past periods
  - get_logs to reference important steps and anomalies
- It then composes the EXECUTIVE BUDGET REPORT with sections:
  - Overview
  - Key Variance Drivers
  - Department Highlights
  - Risks & Opportunities
  - Recommended Follow-up Actions

6. Optional: Orchestrator → Evaluation Agent

- If the user asks things like “How good is the anomaly detector?”,the orchestrator calls the Evaluation Agent.
- This agent uses evaluate_anomaly_detector to compute precision, recall, and F1, and returns an explanation of model quality.

7. Final Output → User

- The orchestrator returns the final EXECUTIVE BUDGET REPORT (final output) to the user in the ADK UI.
- The report is grounded in:
   - tool outputs (numerical analysis),
   - memory (run history), and
   - logs (for transparency).

## 🧩 Features Implemented (Capstone Requirements)

 🎯 1. Multi-Agent System

- 5 specialized LLM agents
- Clear sequential orchestration
- Agent-to-agent delegation

 🎯 2. Tools

- 10+ custom tools
- Data loading, variance computation, anomaly detection
- Memory, logging, evaluation tools

 🎯 3. Memory

- Long-term memory via save_run_summary
- Trends used by reporting agent

 🎯 4. Observability

- Structured logs (log_event)
- Logs retrieved via get_logs()

 🎯 5. Agent Evaluation

- Precision, recall, F1 scoring
- Interpreted by evaluation agent

## 🧪 Evaluation (Precision / Recall / F1)

The project includes a complete evaluation pipeline:
- Ground truth anomalies = variance ≥ 20%
- Predicted anomalies = variance ≥ 15%
Computes: TP, FP, FN, TN, Precision, Recall, F1
Evaluated by evaluation_agent.

## 🧠 Why This Matters

Organizations can adopt this system to:
- Automate monthly financial reviews
- Detect hidden budget issues
- Standardize reporting
- Improve executive decision-making
- Reduce manual analyst workload

This represents a modern AI-native finance workflow.


