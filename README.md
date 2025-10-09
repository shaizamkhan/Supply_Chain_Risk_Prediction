# 🚀 SARIMAX-XGBoost Cascade Model (SXCM) for Supply Chain Risk Mitigation

## Project Domain

Time-Series Forecasting, Supervised Machine Learning, Supply Chain Optimization, Predictive Analytics.

## Overview

This project successfully developed and rigorously validated the **SARIMAX-XGBoost Cascade Model (SXCM)**, a novel, two-stage predictive pipeline meticulously engineered to forecast the **Daily Operational Risk Proportion ($\mathbf{P}_{\text{High\_Risk}}$)** across a complex global supply chain. The architecture processes three years of granular, hourly operational and external data, translating it into a robust, day-ahead risk score. This capability transforms the organization's approach from reactive crisis management to **proactive, data-driven control**, enabling dynamic resource allocation strategies and, crucially, uncovering a profound, long-term structural inefficiency within the supply chain that hinders true resilience.

## 🎯 Key Achievements and Business Impact

| Category | Finding/Achievement | Metric/Value | Strategic Implication |
| :--- | :--- | :--- | :--- |
| **Prediction Accuracy** | Achieved exceptional historical fit and generalization on the unseen test data. The low error confirms the model's reliability for live deployment and integration into existing operational systems. | **RMSE of 0.0273** on $\mathbf{P}_{\text{High\_Risk}}$ (Daily Risk Proportion) | Validated for immediate deployment in tactical planning, specifically for scheduling personnel and managing reserve capacity buffers against predicted volatility. |
| **Diagnostic Insight** | Identified a fixed, internal systemic inefficiency that acts as a performance floor for the entire network. This structural delay is uniform regardless of external conditions like traffic or weather. | $\mathbf{\sim 5.1}$ **hour** structural delay floor, confirmed by isolating trend components. | Mandates aggressive internal process re-engineering (e.g., Lean/Six Sigma) to eliminate this non-value-added "process tax" before any meaningful risk reduction is possible. |
| **Operational State** | Confirmed through multi-year forecasting that the system operates in a state of chronic high risk, a condition driven by internal constraints rather than external factors. This is the **Forecast Paradox**. | Forecasted $\mathbf{\sim 75\%}$ risk baseline (Prediction Paradox). | Justifies the urgent need for building long-term "system slack" or buffer capacity and prioritizes capital expenditure toward solving the internal bottleneck over contingency planning for external events. |
| **Actionable Tool** | Created a highly practical, relative risk classification system to translate the continuous numerical forecast into an easily interpreted operational signal. | **Dynamic Percentile Bucketing (DPB)**, using $25^{\text{th}}$ and $75^{\text{th}}$ percentile thresholds. | Enables targeted, high-value allocation of expensive operational resources only to the **worst $25\%$** of predicted days, significantly improving cost efficiency and focusing management attention. |

## ⚙️ Technical Stack

| Area | Technology/Method | Purpose |
| :--- | :--- | :--- |
| **Modeling (Phase I)** | **SARIMAX** (Seasonal ARIMA w/ Exogenous Variables) | Linear time-series forecasting of Delivery Time Deviation, isolating trend, seasonality, and the temporal impact of exogenous variables for Phase II. |
| **Modeling (Phase II)** | **XGBoost Regressor** | Non-linear ensemble classification of the final risk proportion, using the Phase I forecast as its most valuable input feature alongside lagged operational metrics. |
| **Data Preparation** | **Isolation Forest** | Employed for robust multi-dimensional outlier analysis. Critical failure events ( $<1\%$ of the data) were strategically retained to ensure the model learned to predict high-impact events. |
| **Robustness** | **VIF Filtering** | Ensured the statistical stability and interpretability of the $\text{SARIMAX}$ model by iteratively filtering out multicollinear exogenous variables until the Variance Inflation Factor was below 5. |
| **Validation** | **Chronological Split** | Strict $80/20$ training/testing split ($\text{shuffle=False}$) was used to simulate real-world deployment conditions and prevent look-ahead bias (temporal data leakage). |
| **Libraries** | `statsmodels`, `xgboost`, `sklearn`, `pandas` | Core implementation libraries, chosen for their efficiency and suitability for time-series and gradient boosting tasks. |

## File Structure

| File | Description |
| :--- | :--- |
| `README.md` | This high-impact summary file, detailing project overview and results. |
| `SUMMARY.md` | Detailed technical deep dive into methodology, aggregation rules, and diagnostic findings. |
| `supply_chain_model.py` | Clean, executable Python code demonstrating VIF, SARIMAX, XGBoost, and DPB logic in a single pipeline. |
