# Data Dictionary & Schema Specification: Fund Manager Skill Detector

## 1. Executive Summary
This document provides the exact field schema, data types, units, and transformation formulas for raw and processed datasets in the Fund Manager Skill Detector project.

---

## 2. Processed Analysis Schema (`data/processed/aligned_analysis_dataset.csv`)

| Field Name | Data Type | Units / Format | Description | Formula / Source |
| :--- | :--- | :--- | :--- | :--- |
| `Date` | `datetime64[ns]` | `YYYY-MM-DD` | Common trading date index | Primary Key |
| `symbol` | `string` | Ticker | Active mutual fund symbol | Fund Key |
| `Adj Close` | `float64` | USD ($) | Source Adjusted Close price | Yahoo Finance raw `Adj Close` |
| `daily_return` | `float64` | Decimal | Daily total net return | $\frac{\text{AdjClose}_t - \text{AdjClose}_{t-1}}{\text{AdjClose}_{t-1}}$ |
| `market_adj_close` | `float64` | USD Index Points | S&P 500 Index level | Yahoo Finance raw `^GSPC` |
| `market_return` | `float64` | Decimal | Daily S&P 500 benchmark return | $\frac{\text{SP500}_t - \text{SP500}_{t-1}}{\text{SP500}_{t-1}}$ |
| `Mkt-RF` | `float64` | Decimal | Market risk premium factor | Kenneth French $\text{raw\_Mkt-RF} / 100$ |
| `SMB` | `float64` | Decimal | Small Minus Big factor | Kenneth French $\text{raw\_SMB} / 100$ |
| `HML` | `float64` | Decimal | High Minus Low factor | Kenneth French $\text{raw\_HML} / 100$ |
| `RF` | `float64` | Decimal | Daily risk-free return rate | Kenneth French $\text{raw\_RF} / 100$ |
| `Mom` | `float64` | Decimal | Momentum factor ($UMD$) | Kenneth French $\text{raw\_Mom} / 100$ |
| `excess_fund_return` | `float64` | Decimal | Fund excess return ($Y_t$) | $\text{daily\_return} - RF$ |
| `excess_market_return` | `float64` | Decimal | Market excess return ($X_{1,t}$) | $\text{market\_return} - RF$ |

---

## 3. Screening Metadata Schema (`data/raw/screening_metadata.json`)

| Field Name | Data Type | Description | Current Status |
| :--- | :--- | :--- | :--- |
| `ticker` | `string` | Mutual fund symbol | Recorded |
| `fund_name` | `string` | Full legal title | Recorded |
| `status` | `string` | Selection stage | `"CANDIDATE"` |
| `active_mandate_evidence` | `string / null` | Prospectus mandate excerpt | `null` (Pending) |
| `expense_ratio_evidence` | `float64 / null` | Net Expense Ratio % | `null` (Pending) |
| `active_share_or_differentiation_evidence` | `float64 / null` | Active Share % / Tracking Error | `null` (Pending) |
| `verification_status` | `string` | Verification state | `"UNVERIFIED_CANDIDATE"` |
