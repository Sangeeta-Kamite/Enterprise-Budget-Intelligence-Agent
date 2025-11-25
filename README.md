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


🏗️ Architecture Diagram

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


