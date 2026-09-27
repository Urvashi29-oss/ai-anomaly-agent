# 🛡️ Autonomous AI Anomaly Detection Agent

An intelligent end-to-end AI Agent application that ingests tabular datasets (CSV / Excel), detects complex outliers and statistical anomalies, uses **Google Gemini AI** to synthesize an executive root-cause briefing, dispatches rich **HTML email alerts** via SMTP, and delivers an interactive **visual dashboard** with one-click data exports.

---

## 🚀 Key Features

1. **📁 Multi-Format Dataset Ingestion**:
   - Supports CSV, Excel (`.xlsx`, `.xls`).
   - Built-in one-click realistic datasets for **Financial Fraud Detection** and **Server Telemetry Monitoring**.
   - Automated numeric feature profiling and missing-value imputation.

2. **⚙️ Machine Learning Anomaly Detection**:
   - **Isolation Forest**: Multi-dimensional tree ensemble, robust against high dimensionality and complex feature interactions.
   - **Local Outlier Factor (LOF)**: Density-based local neighborhood anomaly detector.
   - **Z-Score (Statistical)**: Multivariate extreme value tail detector.
   - **Ensemble Mode**: Combined Isolation Forest + Statistical Z-score.
   - **Feature Attribution & Severity Ranking**: Identifies the primary driver behind every detected anomaly and ranks them as Critical, Medium, or Low risk.

3. **🧠 AI Agent Executive Summarization**:
   - Powered by **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`).
   - Generates executive briefings, root cause analysis, and prioritized mitigation steps.
   - Built-in offline rule intelligence fallback if no API key is provided.

4. **📧 SMTP Email Alert Dispatcher**:
   - Preset support for **Gmail**, **Outlook / Office 365**, and **Custom SMTP** servers.
   - Dispatches responsive **HTML email alerts** with color-coded severity badges, KPI metric tiles, and anomaly tables.
   - Includes automatic CSV attachment of all flagged outliers.
   - In-app interactive email preview and connection tester.

5. **📊 Interactive Visual Dashboard**:
   - 2D scatter plots of any feature pair with color-coded severity.
   - Continuous anomaly score distribution histogram.
   - Time-series anomaly timeline (auto-detected from timestamp/date columns).
   - Baseline vs Anomaly cohort deviation comparison table.
   - One-click CSV and Markdown report downloads.



