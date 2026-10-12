"""
report_generator.py — Phase 4: Institutional PDF Tear-Sheet Generator
=======================================================================
Generates a one-page Institutional Due Diligence Tear-Sheet for any fund
analysed by the skill detector engine.

Uses Jinja2 for HTML templating + pdfplumber-free path (built-in html approach).
Output: bytes of a styled PDF ready for st.download_button().

Usage:
    from components.report_generator import generate_pdf_bytes, build_fund_data
"""

from __future__ import annotations

import io
import base64
from datetime import datetime
from typing import Optional

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Category colours (must match combined_data.CATEGORY_INFO)
CATEGORY_COLORS = {
    "Strong Evidence of Skill":              "#166534",
    "Strong Skill Candidate (Preliminary)":  "#059669",
    "Moderate Evidence":                     "#2563eb",
    "Positive but Inconclusive":             "#0ea5e9",
    "Weak/Inconclusive":                     "#64748b",
    "Likely Style/Luck Driven":              "#d97706",
    "Unclassified":                          "#94a3b8",
}

# ─── HTML template ─────────────────────────────────────────────────────────────
_TEMPLATE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    padding: 36px 44px;
    font-size: 13px;
    line-height: 1.5;
  }

  .header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e4d8c 60%, #0c2461 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 10px;
    margin-bottom: 18px;
  }
  .header h1 { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
  .header .sub { font-size: 11.5px; opacity: 0.82; }
  .header .tag { font-size: 10.5px; opacity: 0.65; margin-top: 6px; }

  .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    color: white;
    background: {{ badge_color }};
    margin: 6px 0;
  }

  .score-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 10px 0 16px 0;
  }
  .score-circle {
    width: 56px; height: 56px;
    border-radius: 50%;
    border: 3px solid {{ badge_color }};
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; font-weight: 700; color: {{ badge_color }};
    flex-shrink: 0;
  }
  .score-label { font-size: 11px; color: #64748b; }

  .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 14px 0;
  }
  .metric-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
    background: #f8fafc;
  }
  .metric-val { font-size: 18px; font-weight: 700; color: #1e3a8a; }
  .metric-lbl { font-size: 9.5px; color: #64748b; text-transform: uppercase;
                letter-spacing: 0.04em; margin-top: 2px; }

  .section-title {
    font-size: 11px; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.07em;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin: 16px 0 8px 0;
  }

  table.data-table {
    width: 100%; border-collapse: collapse; font-size: 11.5px;
  }
  table.data-table th {
    text-align: left; padding: 5px 8px;
    background: #f1f5f9; border-bottom: 2px solid #e2e8f0;
    font-weight: 600; color: #475569; font-size: 10.5px;
    text-transform: uppercase;
  }
  table.data-table td {
    padding: 5px 8px; border-bottom: 1px solid #f1f5f9;
  }
  table.data-table tr:last-child td { border-bottom: none; }
  .pos { color: #166534; font-weight: 600; }
  .neg { color: #dc2626; font-weight: 600; }

  .explanation {
    background: #f0f9ff; border-left: 4px solid {{ badge_color }};
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 12px; margin: 10px 0;
  }

  .flags-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 8px 0;
  }
  .flag-item {
    display: flex; align-items: center; gap: 8px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 6px 10px; font-size: 11px;
  }
  .flag-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .flag-pass { background: #10b981; }
  .flag-fail { background: #ef4444; }
  .flag-neutral { background: #94a3b8; }

  .footer {
    margin-top: 24px; padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    font-size: 9.5px; color: #94a3b8;
    display: flex; justify-content: space-between;
  }

  .watermark {
    text-align: center; font-size: 9px; color: #cbd5e1;
    margin-top: 8px;
  }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>Institutional Due Diligence Tear-Sheet</h1>
  <div class="sub">{{ fund_name }}</div>
  <div class="tag">
    Evaluation Rule Set v{{ eval_version }} &nbsp;|&nbsp;
    Analysis Period: {{ period }} &nbsp;|&nbsp;
    Market: {{ market }} &nbsp;|&nbsp;
    Observations: {{ n_obs }}
  </div>
</div>

<!-- EVIDENCE CLASS + SCORE -->
<div class="score-row">
  <div class="score-circle">{{ score }}</div>
  <div>
    <div style="font-size:11px;color:#64748b;margin-bottom:2px;">Evidence Classification</div>
    <span class="badge">{{ evidence_class }}</span>
    <div class="score-label">Score: {{ score }}/100 &nbsp;|&nbsp; Evaluation version: {{ eval_version }}</div>
  </div>
</div>

<!-- EXPLANATION -->
<div class="explanation">{{ explanation }}</div>

<!-- KEY METRICS -->
<div class="section-title">Core Statistical Results</div>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-val {{ 'pos' if capm_alpha_sign == '+' else 'neg' }}">{{ capm_alpha }}</div>
    <div class="metric-lbl">CAPM Alpha (% p.a.)</div>
  </div>
  <div class="metric-card">
    <div class="metric-val {{ 'pos' if factor_alpha_sign == '+' else 'neg' }}">{{ factor_alpha }}</div>
    <div class="metric-lbl">4-Factor Alpha (% p.a.)</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{{ capm_p }}</div>
    <div class="metric-lbl">CAPM HAC p-value</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{{ bootstrap_p }}</div>
    <div class="metric-lbl">Bootstrap p-value</div>
  </div>
</div>

<!-- DETAILED RESULTS TABLE -->
<div class="section-title">Detailed Pillar Results</div>
<table class="data-table">
  <thead>
    <tr>
      <th>Pillar</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Threshold</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CAPM (Newey-West HAC)</td>
      <td>Annual Alpha</td>
      <td class="{{ 'pos' if capm_alpha_sign == '+' else 'neg' }}">{{ capm_alpha }}</td>
      <td>&gt; 0%</td>
      <td>{{ "✓ Positive" if capm_alpha_sign == "+" else "✗ Non-positive" }}</td>
    </tr>
    <tr>
      <td>CAPM (Newey-West HAC)</td>
      <td>HAC p-value</td>
      <td>{{ capm_p }}</td>
      <td>&lt; 0.05</td>
      <td>{{ "✓ Significant" if capm_significant else "✗ Not significant" }}</td>
    </tr>
    <tr>
      <td>Block Bootstrap (B={{ n_bootstrap }})</td>
      <td>Bootstrap p-value</td>
      <td>{{ bootstrap_p }}</td>
      <td>&lt; 0.05</td>
      <td>{{ "✓ Significant" if bootstrap_significant else "✗ Not significant" }}</td>
    </tr>
    <tr>
      <td>Carhart 4-Factor</td>
      <td>4-Factor Alpha</td>
      <td class="{{ 'pos' if factor_alpha_sign == '+' else 'neg' }}">{{ factor_alpha }}</td>
      <td>&gt; 0%, p &lt; 0.05</td>
      <td>{{ "✓ Survives" if factor_survives else "✗ Does not survive" }}</td>
    </tr>
    <tr>
      <td>Intertemporal Persistence</td>
      <td>Positive subperiods</td>
      <td>{{ positive_periods }} of {{ total_periods }}</td>
      <td>&gt; 50% (strict majority)</td>
      <td>{{ "✓ Persistent" if persistent else "✗ Not persistent" }}</td>
    </tr>
    <tr>
      <td>Regression Fit</td>
      <td>Beta (market sensitivity)</td>
      <td>{{ beta_val }}</td>
      <td>—</td>
      <td>{{ "Defensive (β &lt; 1)" if beta_lt1 else "Aggressive (β ≥ 1)" }}</td>
    </tr>
  </tbody>
</table>

<!-- FACTOR LOADINGS -->
<div class="section-title">4-Factor Loadings (Carhart Model)</div>
<table class="data-table">
  <thead>
    <tr>
      <th>Factor</th>
      <th>Loading (β)</th>
      <th>t-stat</th>
      <th>p-value</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Market (Mkt-RF)</td>
      <td>{{ factor_beta_mkt }}</td>
      <td>{{ factor_t_mkt }}</td>
      <td>{{ factor_p_mkt }}</td>
      <td>Market exposure</td>
    </tr>
    <tr>
      <td>Size (SMB)</td>
      <td>{{ factor_beta_smb }}</td>
      <td>{{ factor_t_smb }}</td>
      <td>{{ factor_p_smb }}</td>
      <td>{{ "Small-cap tilt" if factor_beta_smb_pos else "Large-cap tilt" }}</td>
    </tr>
    <tr>
      <td>Value (HML)</td>
      <td>{{ factor_beta_hml }}</td>
      <td>{{ factor_t_hml }}</td>
      <td>{{ factor_p_hml }}</td>
      <td>{{ "Value tilt" if factor_beta_hml_pos else "Growth tilt" }}</td>
    </tr>
    <tr>
      <td>Momentum (WML)</td>
      <td>{{ factor_beta_mom }}</td>
      <td>{{ factor_t_mom }}</td>
      <td>{{ factor_p_mom }}</td>
      <td>{{ "Momentum tilt" if factor_beta_mom_pos else "Contrarian tilt" }}</td>
    </tr>
  </tbody>
</table>

<!-- FOOTER -->
<div class="footer">
  <div>Generated: {{ generated_at }} &nbsp;|&nbsp; Fund Manager Skill Detector — v{{ eval_version }}</div>
  <div>Statistical evidence only — not investment advice</div>
</div>
<div class="watermark">github.com/your-repo/fund-manager-skill-detector</div>

</body>
</html>
"""


# ─── Data builder ──────────────────────────────────────────────────────────────

def build_report_data(fund_name: str,
                       market: str,
                       result: dict,
                       n_bootstrap: int = 500) -> dict:
    """
    Builds the template context dict from a live pipeline result dict.

    Args:
        fund_name:   Display name for the fund
        market:      "India" or "USA"
        result:      Output dict from live_pipeline.run_india_live() or run_usa_live()
        n_bootstrap: Number of bootstrap iterations used
    """
    capm    = result.get("capm", {})
    boot    = result.get("bootstrap", {})
    factor  = result.get("factor", {})
    pers    = result.get("persistence", {})
    ev      = result.get("evaluation", {})

    def _pct(v, digits=2):
        if v is None or (isinstance(v, float) and (v != v)):
            return "n/a"
        return f"{float(v):+.{digits}f}%"

    def _p(v):
        if v is None or (isinstance(v, float) and (v != v)):
            return "n/a"
        return "<0.001" if float(v) < 0.001 else f"{float(v):.3f}"

    def _f(v, digits=3):
        if v is None or (isinstance(v, float) and (v != v)):
            return "n/a"
        return f"{float(v):.{digits}f}"

    capm_alpha_annual = capm.get("alpha_annual_pct", 0.0) or 0.0
    factor_alpha_pct  = (factor.get("alpha_daily", 0.0) or 0.0) * 252 * 100
    capm_p_val        = capm.get("alpha_pvalue_hac", 1.0) or 1.0
    boot_p_val        = boot.get("bootstrap_pvalue", 1.0) or 1.0
    beta              = capm.get("beta", 1.0) or 1.0

    pos_periods   = int(pers.get("positive_alpha_periods", 0) or 0)
    total_periods = int(pers.get("total_periods", 4) or 4)

    f_beta_mkt = factor.get("mkt_rf_coef", 0.0) or 0.0
    f_beta_smb = factor.get("smb_coef", 0.0)    or 0.0
    f_beta_hml = factor.get("hml_coef", 0.0)    or 0.0
    f_beta_mom = factor.get("mom_coef", 0.0)    or 0.0
    f_t_mkt    = factor.get("mkt_rf_tstat", 0.0) or 0.0
    f_t_smb    = factor.get("smb_tstat", 0.0)    or 0.0
    f_t_hml    = factor.get("hml_tstat", 0.0)    or 0.0
    f_t_mom    = factor.get("mom_tstat", 0.0)    or 0.0
    f_p_mkt    = factor.get("mkt_rf_pvalue", 1.0) or 1.0
    f_p_smb    = factor.get("smb_pvalue", 1.0)    or 1.0
    f_p_hml    = factor.get("hml_pvalue", 1.0)    or 1.0
    f_p_mom    = factor.get("mom_pvalue", 1.0)    or 1.0

    period_str = f"{capm.get('start_date', '2019-01-01')} — {capm.get('end_date', '2023-12-31')}"

    return {
        "fund_name":          fund_name,
        "market":             market,
        "eval_version":       ev.get("evaluation_version", "2.3.0"),
        "period":             period_str,
        "n_obs":              result.get("n_obs", "n/a"),
        "n_bootstrap":        n_bootstrap,
        "evidence_class":     ev.get("evidence_class", "Unclassified"),
        "score":              ev.get("score", 0),
        "explanation":        ev.get("explanation", "No explanation available."),
        "badge_color":        CATEGORY_COLORS.get(ev.get("evidence_class", "Unclassified"), "#94a3b8"),

        # Core metrics
        "capm_alpha":         _pct(capm_alpha_annual),
        "capm_alpha_sign":    "+" if capm_alpha_annual >= 0 else "-",
        "factor_alpha":       _pct(factor_alpha_pct),
        "factor_alpha_sign":  "+" if factor_alpha_pct >= 0 else "-",
        "capm_p":             _p(capm_p_val),
        "bootstrap_p":        _p(boot_p_val),
        "beta_val":           _f(beta),
        "beta_lt1":           float(beta) < 1.0,

        # Status flags
        "capm_significant":    float(capm_p_val) < 0.05 and capm_alpha_annual > 0,
        "bootstrap_significant": float(boot_p_val) < 0.05,
        "factor_survives":     ev.get("factor_alpha_survives", False),
        "persistent":          pos_periods > total_periods / 2,
        "positive_periods":    pos_periods,
        "total_periods":       total_periods,

        # Factor loadings
        "factor_beta_mkt":     _f(f_beta_mkt),
        "factor_t_mkt":        _f(f_t_mkt),
        "factor_p_mkt":        _p(f_p_mkt),
        "factor_beta_smb":     _f(f_beta_smb),
        "factor_t_smb":        _f(f_t_smb),
        "factor_p_smb":        _p(f_p_smb),
        "factor_beta_hml":     _f(f_beta_hml),
        "factor_t_hml":        _f(f_t_hml),
        "factor_p_hml":        _p(f_p_hml),
        "factor_beta_mom":     _f(f_beta_mom),
        "factor_t_mom":        _f(f_t_mom),
        "factor_p_mom":        _p(f_p_mom),
        "factor_beta_smb_pos": float(f_beta_smb) > 0,
        "factor_beta_hml_pos": float(f_beta_hml) > 0,
        "factor_beta_mom_pos": float(f_beta_mom) > 0,

        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ─── PDF generator ─────────────────────────────────────────────────────────────

def generate_html_report(data: dict) -> str:
    """Renders the HTML tear-sheet. Returns the HTML string."""
    if not JINJA2_AVAILABLE:
        return "<html><body><p>Jinja2 not installed. Run: pip install jinja2</p></body></html>"
    return Template(_TEMPLATE_HTML).render(**data)


def generate_pdf_bytes(data: dict) -> Optional[bytes]:
    """
    Renders the HTML template and converts to PDF bytes using weasyprint.
    Returns None if weasyprint is not installed (falls back to HTML download).
    """
    html_str = generate_html_report(data)
    try:
        import weasyprint
        return weasyprint.HTML(string=html_str).write_pdf()
    except ImportError:
        # weasyprint not available — return the HTML as bytes instead
        return html_str.encode("utf-8")
    except Exception:
        return html_str.encode("utf-8")
