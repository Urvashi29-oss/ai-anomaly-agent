import os
import json
from typing import Dict, Any, Optional

# Try new google-genai SDK first, then legacy google.generativeai
HAS_NEW_GENAI = False
HAS_LEGACY_GENAI = False

try:
    from google import genai
    from google.genai import types
    HAS_NEW_GENAI = True
except ImportError:
    pass

if not HAS_NEW_GENAI:
    try:
        import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        pass

class AnomalyAIAgent:
    """
    AI Agent that analyzes anomaly detection results using Google Gemini,
    extracts actionable insights, and crafts professional alert notifications.
    Supports both the modern google-genai SDK and legacy google-generativeai.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None
        self.client_ready = False
        
        if self.api_key:
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    self.client_ready = True
                except Exception as e:
                    print(f"Error initializing new google.genai client: {e}")
            
            if not self.client_ready and HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self.client_ready = True
                except Exception as e:
                    print(f"Error configuring legacy google.generativeai: {e}")

    def generate_summary(self, summary_metrics: Dict[str, Any], sample_anomalies: list) -> Dict[str, str]:
        """
        Generates comprehensive AI analysis, executive summary, and alert messages.
        """
        if self.client_ready:
            try:
                return self._call_gemini_analysis(summary_metrics, sample_anomalies)
            except Exception as e:
                print(f"Gemini call error: {e}. Falling back to rule-based agent analysis.")
                return self._generate_fallback_summary(summary_metrics, sample_anomalies, error_msg=str(e))
        else:
            return self._generate_fallback_summary(summary_metrics, sample_anomalies)

    def _call_gemini_analysis(self, summary_metrics: Dict[str, Any], sample_anomalies: list) -> Dict[str, str]:
        """Calls Google Gemini model to summarize the anomalies."""
        prompt = f"""
You are an expert Autonomous AI Data Quality & Anomaly Detection Agent.
You have just analyzed a dataset and detected statistical and machine learning anomalies.

Here is the statistical summary of the detection:
{json.dumps(summary_metrics, indent=2)}

Here are sample detected anomalous records (up to 5 most severe):
{json.dumps(sample_anomalies[:5], indent=2, default=str)}

Respond with a JSON object ONLY (no markdown fences, pure JSON) with the following structure:
{{
  "executive_summary": "High-level 2-3 paragraph executive summary of findings, data health, and risk assessment.",
  "root_causes": "Bullet points detailing likely root causes and which specific features contributed most to the anomalies.",
  "recommendations": "Actionable, numbered list of recommendations for the operations/engineering/business team.",
  "email_subject": "A concise, high-priority email alert subject line (e.g., '[ALERT] X Anomalies Detected in Dataset - High Severity')",
  "email_body": "A professionally formatted plain text or markdown email body ready to be sent to stakeholders detailing the anomalies and urgent next steps."
}}
"""
        text = ""
        # 1. Try modern google-genai client
        if HAS_NEW_GENAI and self.client:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            text = response.text.strip()
        # 2. Try legacy google-generativeai
        elif HAS_LEGACY_GENAI:
            model = legacy_genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            text = response.text.strip()
        else:
            raise RuntimeError("No Gemini SDK available.")
        
        # Clean potential markdown fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            return {
                "executive_summary": parsed.get("executive_summary", ""),
                "root_causes": parsed.get("root_causes", ""),
                "recommendations": parsed.get("recommendations", ""),
                "email_subject": parsed.get("email_subject", f"🚨 Anomaly Alert: {summary_metrics['anomaly_count']} outliers detected"),
                "email_body": parsed.get("email_body", ""),
                "source": f"Google Gemini ({self.model_name})"
            }
        except json.JSONDecodeError:
            return {
                "executive_summary": text,
                "root_causes": "Refer to executive summary above.",
                "recommendations": "Investigate flagged records with high anomaly scores.",
                "email_subject": f"🚨 Anomaly Alert: {summary_metrics['anomaly_count']} anomalies detected ({summary_metrics['anomaly_percentage']}%)",
                "email_body": text,
                "source": f"Google Gemini ({self.model_name})"
            }

    def _generate_fallback_summary(self, summary_metrics: Dict[str, Any], sample_anomalies: list, error_msg: Optional[str] = None) -> Dict[str, str]:
        """Intelligent rule-based fallback summary when API key is not configured or offline."""
        total = summary_metrics.get("total_records", 0)
        count = summary_metrics.get("anomaly_count", 0)
        pct = summary_metrics.get("anomaly_percentage", 0.0)
        critical = summary_metrics.get("critical_count", 0)
        method = summary_metrics.get("method_used", "Statistical ML")
        top_factors = summary_metrics.get("top_affected_columns", {})

        top_factors_str = ", ".join([f"{k} ({v} occurrences)" for k, v in list(top_factors.items())[:3]]) if top_factors else "various numeric attributes"

        if pct > 10 or critical > 5:
            risk_level = "CRITICAL / HIGH"
            risk_note = "Elevated outlier density indicates potential systemic malfunction, fraud surge, or data collection integrity breakdown."
        elif pct > 4:
            risk_level = "MODERATE"
            risk_note = "Moderate anomalies detected. Notable variance from standard baseline operating metrics."
        else:
            risk_level = "LOW / NORMAL FLUCTUATIONS"
            risk_note = "Anomalies are isolated and represent standard tail-distribution variance."

        exec_summary = (
            f"### Executive Anomaly Brief\n\n"
            f"- **Dataset Scope**: {total:,} total records evaluated using **{method}**.\n"
            f"- **Outliers Identified**: **{count:,} records ({pct}%)** met outlier criteria.\n"
            f"- **Critical Outliers**: **{critical:,} records** scored in the high-severity tier.\n"
            f"- **Overall Risk Level**: **{risk_level}**.\n\n"
            f"{risk_note}\n\n"
            f"The primary variance drivers identified across the flagged records were: **{top_factors_str}**."
        )

        root_causes = (
            f"- **Key Deviation Factor**: The most frequent driver for flagged anomalies is `{list(top_factors.keys())[0] if top_factors else 'feature distribution'}`.\n"
            f"- **Multivariate Shifts**: Significant cross-feature deviation observed between normal baseline averages and anomaly cohorts.\n"
            f"- **Potential Etiologies**: Sensor drift, fraudulent/unusual transaction spikes, extreme user behavior, or input formatting discrepancies."
        )

        recommendations = (
            f"1. **Triage Critical Rows**: Immediately inspect the top {min(critical, 10)} records with anomaly scores > 0.80.\n"
            f"2. **Feature Deep-Dive**: Validate integrity and pipeline ingestion for primary driver: `{list(top_factors.keys())[0] if top_factors else 'key fields'}`.\n"
            f"3. **Stakeholder Notification**: Dispatch the automated email alert to operational and engineering teams for follow-up.\n"
            f"4. **Threshold Calibration**: Adjust algorithm contamination settings if historical tolerance differs from {summary_metrics.get('contamination_rate', 0.05)*100}%."
        )

        email_subj = f"[{risk_level} ALERT] {count} Anomalies Detected in Dataset ({pct}% Outlier Ratio)"

        email_body = (
            f"ANOMALY DETECTION SYSTEM ALERT\n"
            f"=========================================\n\n"
            f"Detection Algorithm : {method}\n"
            f"Total Records       : {total:,}\n"
            f"Anomalies Found     : {count:,} ({pct}%)\n"
            f"Critical Outliers   : {critical:,}\n"
            f"Risk Assessment     : {risk_level}\n"
            f"Primary Drivers     : {top_factors_str}\n\n"
            f"RECOMMENDED IMMEDIATE ACTIONS:\n"
            f"1. Review the attached anomaly report and verify high-severity records.\n"
            f"2. Check data pipeline for {top_factors_str}.\n"
            f"3. Acknowledge and resolve alerts in the operations dashboard.\n\n"
            f"Generated automatically by AI Anomaly Detection Agent."
        )

        source_info = "Built-in Rule Engine (Configure Gemini API Key in sidebar for AI synthesis)"
        if error_msg:
            source_info += f" [Note: Gemini API returned: {error_msg}]"

        return {
            "executive_summary": exec_summary,
            "root_causes": root_causes,
            "recommendations": recommendations,
            "email_subject": email_subj,
            "email_body": email_body,
            "source": source_info
        }
