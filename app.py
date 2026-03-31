from __future__ import annotations
# -*- coding: utf-8 -*-
# Profit Mix Optimizer — v4.0
# ארכיטקטורה: app.py → loader.py + optimizer.py + ui_components.py

import itertools, math, os, re, html, io, traceback
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loader import (
    FUNDS_GSHEET_ID, POLICIES_GSHEET_ID, PENSION_GSHEET_ID,
    GEMEL_GSHEET_ID, GEMEL_INV_GSHEET_ID, SERVICE_GSHEET_ID,
    load_funds_long, _gsheet_to_bytes, _load_service_scores,
    _extract_manager, _extract_manager_policy, _match_param,
    parse_clearing_report, _compute_baseline_from_holdings,
    PARAM_ALIASES, _POLICY_SUB_MGR, _POLICY_INSURER_PREFIXES, _POLICY_DIRECT,
)
from optimizer import (
    find_best_solutions, _pick_recommendations, _pick_three_distinct,
    _weights_for_n, _prefilter_candidates, _hard_ok_vec,
    _weights_items, _weights_short, _make_advantage,
    _normalize_series, _weighted_metric_from_items,
    _compute_weighted_returns_for_items, _manager_weights_from_items,
    _change_type_badge,
)
from ui_components import (
    QUICK_PROFILES_NEW, _DEFAULT_UNCHECKED_PATTERNS, _is_default_unchecked,
    render_product_selector, render_header, render_quick_filters,
    render_mix_builder, render_best_solution, render_results_strip,
    render_results_table, render_fund_comparison, render_history,
    _ai_explain, _export_excel, _alloc_plot, _manager_donut, _radar_chart,
    _render_compact_card, _kpi_chip_html, _delta_grid_html,
    _mini_alloc_bar_html, _lbl, _pct, _num, _chip,
)

st.set_page_config(
    page_title="Profit Mix Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import streamlit as _st_check
_st_version = tuple(int(x) for x in _st_check.__version__.split(".")[:2])

def _safe_plotly(fig, key=None):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        try:
            st.plotly_chart(fig, key=key)
        except TypeError:
            st.plotly_chart(fig)

# ─────────────────────────────────────────────
# CSS – Premium Fintech (v3.6)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base RTL ── */
html, body, [class*="css"] {
  direction: rtl; text-align: right;
  font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
  background: #F5F7FA;
}
div[data-baseweb="slider"], div[data-baseweb="slider"] * { direction: ltr !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 3.8rem 1.2rem 3rem !important; max-width: 1180px; }
div[data-testid="stVerticalBlock"] { gap: 0.25rem; }
/* tighten metric spacing */
div[data-testid="stMetric"] { padding: 4px 0 !important; }
div[data-testid="stMetricLabel"] { font-size: 11px !important; color: #6B7280 !important; }
div[data-testid="stMetricValue"] { font-size: 16px !important; font-weight: 800 !important; }

/* ── Header ── */
.pmo-header {
  background: #1F3A5F; color: #fff;
  border-radius: 12px; padding: 14px 20px 12px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 10px; margin-bottom: 8px;
}
.pmo-header-left { display: flex; align-items: center; gap: 12px; }
.pmo-logo-box {
  width: 38px; height: 38px; border-radius: 9px;
  background: #3A7AFE; display: flex; align-items: center;
  justify-content: center; font-size: 19px; flex-shrink: 0;
}
.pmo-title   { font-size: 20px; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
.pmo-sub     { font-size: 11px; color: #93b4e0; margin: 1px 0 0; }
.pmo-kpis    { display: flex; gap: 10px; }
.pmo-kpi     { text-align: center; }
.pmo-kpi-val { font-size: 17px; font-weight: 800; color: #7dd3fc; display: block; line-height: 1.1; }
.pmo-kpi-lbl { font-size: 10px; color: #93b4e0; white-space: nowrap; }

/* ── Quick filters ── */
.qf-wrap {
  display: flex; gap: 6px; flex-wrap: wrap;
  padding: 8px 0 6px; margin-bottom: 6px;
}
/* Streamlit radio styled as pills */
.nav-bar div[role="radiogroup"] {
  display: flex !important; gap: 8px !important; flex-wrap: wrap !important;
}
.nav-bar div[role="radiogroup"] label {
  display: inline-flex !important; align-items: center !important;
  padding: 7px 18px !important; border-radius: 999px !important;
  border: 2px solid #D1D5DB !important; background: #fff !important;
  font-size: 14px !important; font-weight: 700 !important;
  cursor: pointer !important; white-space: nowrap !important;
  color: #374151 !important; transition: all 0.18s !important; margin: 0 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.nav-bar div[role="radiogroup"] label:hover {
  background: #EFF6FF !important; border-color: #3A7AFE !important;
  color: #1F3A5F !important; transform: translateY(-1px) !important;
  box-shadow: 0 3px 8px rgba(58,122,254,0.2) !important;
}
.nav-bar div[role="radiogroup"] label:has(input:checked) {
  background: #1F3A5F !important; color: #ffffff !important;
  border-color: #1F3A5F !important;
  box-shadow: 0 3px 10px rgba(31,58,95,0.35) !important;
  transform: translateY(-1px) !important;
}
.nav-bar div[role="radiogroup"] input[type="radio"] { display: none !important; }
.nav-bar div[role="radiogroup"] p { margin: 0 !important; }

/* ── Cards ── */
.card {
  background: #fff; border: 1px solid #E5EAF2; border-radius: 12px;
  padding: 16px 18px 14px; margin-bottom: 10px;
}
.card-title {
  font-size: 14px; font-weight: 800; color: #111827;
  margin: 0 0 12px; display: flex; align-items: center; gap: 6px;
}
.card-sub { font-size: 11px; color: #6B7280; margin-bottom: 8px; }

/* ── Mix builder ── */
.mix-total {
  background: #F0F4FF; border: 1px solid #C7D7FF;
  border-radius: 8px; padding: 8px 12px; margin: 8px 0;
  font-size: 12px; display: flex; justify-content: space-between;
}
.mix-total .t-ok  { color: #15A46E; font-weight: 800; }
.mix-total .t-warn{ color: #B7791F; font-weight: 800; }
.mix-total .t-err { color: #DC2626; font-weight: 800; }

/* ── Best result card ── */
.br-score {
  font-size: 42px; font-weight: 900; color: #1F3A5F;
  line-height: 1; margin: 0;
}
.br-score-lbl { font-size: 11px; color: #6B7280; margin-top: 2px; }
.br-managers  { font-size: 12px; color: #374151; margin: 8px 0 4px; }
.br-tracks    { font-size: 11px; color: #6B7280; margin-bottom: 10px; }
.br-chips     { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 10px; }
.br-chip {
  padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;
  background: #EFF6FF; color: #1F3A5F; border: 1px solid #C7D7FF;
}

/* ── Results strip ── */
.res-strip {
  background: #F0F4FF; border: 1px solid #C7D7FF;
  border-radius: 8px; padding: 7px 14px;
  font-size: 12px; color: #374151; margin: 6px 0;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
.res-strip b { color: #1F3A5F; }

/* ── Results table ── */
.res-tbl {
  width: 100%; border-collapse: collapse;
  font-size: 12.5px; direction: rtl;
}
.res-tbl th {
  background: #1F3A5F; color: #F0F4FF;
  padding: 7px 10px; text-align: right;
  white-space: nowrap; font-weight: 700; font-size: 11.5px;
}
.res-tbl td {
  padding: 7px 10px; border-bottom: 1px solid #F1F5F9;
  text-align: right; vertical-align: middle;
}
.res-tbl tr.sel-row td { background: #EFF6FF !important; }
.res-tbl tr:hover td  { background: #F9FAFB; }
.res-tbl td.num  { text-align: center; }
.res-tbl td.name { font-weight: 700; color: #111827; }

/* ── Stats bars ── */
.stat-bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; direction: rtl; }
.stat-bar     { height: 14px; border-radius: 3px; min-width: 3px; }

/* ── Misc ── */
.score-tip {
  background: #FFFBEB; border: 1px solid #FDE68A;
  border-radius: 8px; padding: 7px 11px; font-size: 11.5px; color: #78350F; margin: 5px 0;
}
.pw-wrap  { max-width: 300px; margin: 60px auto; text-align: center; }
.pw-title { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
.pw-sub   { font-size: 12px; opacity: 0.7; margin-bottom: 14px; }
.pw-warn  { font-size: 11px; color: #B45309; background: #FEF3C7; border-radius: 6px; padding: 4px 9px; margin-top: 7px; }
.change-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.change-low   { background: #DCFCE7; color: #166534; }
.change-med   { background: #FEF3C7; color: #92400E; }
.change-high  { background: #FEE2E2; color: #991B1B; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .pmo-header { flex-direction: column; align-items: flex-start; }
  .pmo-kpis   { flex-wrap: wrap; }
}
/* dark mode */
@media (prefers-color-scheme: dark) {
  html, body, [class*="css"] { background: #0d1117; }
  .card { background: #161b22; border-color: #30363d; }
  .card-title { color: #f0f6ff; }
  .res-tbl td { border-color: #1e293b; color: #cbd5e1; }
  .res-tbl th { background: #1e3a8a; }
  .res-strip, .mix-total { background: #0f1e3d; border-color: #1e3a5f; color: #cbd5e1; }
  .br-chip { background: #0f1e3d; border-color: #1e3a5f; color: #93c5fd; }
  .score-tip { background: #451a03; border-color: #92400e; color: #fde68a; }
}
div[data-testid="stDataFrame"] * { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _esc(x) -> str:
    try:
        return html.escape("" if x is None else str(x), quote=True)
    except Exception:
        return ""

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[^\d.\-]", "", str(x).replace(",", "").replace("−", "-"))
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _fmt_pct(x, decimals=2) -> str:
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "—"

def _fmt_num(x, fmt="{:.2f}") -> str:
    try:
        return fmt.format(float(x))
    except Exception:
        return "—"


# ─────────────────────────────────────────────
# Password Gate
# ─────────────────────────────────────────────
def _check_password() -> bool:
    if st.session_state.get("auth_ok", False):
        return True
    is_default = True
    if hasattr(st, "secrets") and "APP_PASSWORD" in st.secrets:
        correct = str(st.secrets["APP_PASSWORD"])
        is_default = False
    else:
        correct = os.getenv("APP_PASSWORD", "1234")

    st.markdown("""
    <div class="pw-wrap">
      <div class="pw-title">🔒 כניסה</div>
      <div class="pw-sub">האפליקציה מוגנת בסיסמה</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("סיסמה", type="password", placeholder="••••••••", label_visibility="collapsed")
        if st.button("כניסה", use_container_width=True, type="primary"):
            if pwd == correct:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("סיסמה שגויה")
        if is_default:
            st.markdown(
                '<div class="pw-warn">⚠️ סיסמה ברירת מחדל (1234). הגדר APP_PASSWORD ב-Streamlit Secrets בייצור!</div>',
                unsafe_allow_html=True
            )
    st.stop()

_check_password()


# ─────────────────────────────────────────────
# Google Sheets – מקורות נתונים
# ─────────────────────────────────────────────
FUNDS_GSHEET_ID    = "1bO-Mdvz5sWw73J-5msGdeleK-F4csklG"
POLICIES_GSHEET_ID = "15XIxQxBU4Mfun-rgcwK0Mfq6blUgekRx"
PENSION_GSHEET_ID  = "1EOBY0L2IVUyY8zBYBMKEBkBCixaD9v7m"
GEMEL_GSHEET_ID    = "1JwKWUPj5TxMGbfQwABdZP0XIQOCvQHz9"
GEMEL_INV_GSHEET_ID = "1gvA13OkDHBkf0QjJZQ_Z21jF8bNgAHgJ"
SERVICE_GSHEET_ID  = "1FSgvIG6VsJxB5QPY6fmwAwGc1TYLB0KXg-7ckkD_RJQ"

# ─────────────────────────────────────────────
# Voting – Google Sheets via Service Account
# ─────────────────────────────────────────────
VOTES_SHEET_NAME = "votes"

def _get_votes_worksheet():
    """Return the gspread worksheet for votes, or None if not configured."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        # Credentials stored in Streamlit Secrets as [gcp_service_account]
        if not (hasattr(st, "secrets") and "gcp_service_account" in st.secrets):
            return None
        sa_info = dict(st.secrets["gcp_service_account"])
        creds   = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client  = gspread.authorize(creds)
        sheet   = client.open_by_key(FUNDS_GSHEET_ID)
        try:
            ws = sheet.worksheet(VOTES_SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = sheet.add_worksheet(title=VOTES_SHEET_NAME, rows=2000, cols=8)
            ws.append_row(["timestamp","alternative","managers","tracks",
                           "n_funds","mix_policy","session_hash"], value_input_option="RAW")
        return ws
    except Exception as _e:
        return None


def _write_vote(alternative: str, managers: str, tracks: str) -> bool:
    """Write a single vote row. Returns True on success."""
    try:
        ws = _get_votes_worksheet()
        if ws is None:
            return False
        import hashlib, uuid
        session_id = st.session_state.get("_session_id")
        if not session_id:
            session_id = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:10]
            st.session_state["_session_id"] = session_id
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alternative,
            managers,
            tracks,
            str(st.session_state.get("n_funds", 2)),
            str(st.session_state.get("mix_policy", "")),
            session_id,
        ], value_input_option="RAW")
        return True
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def _load_votes_cached() -> pd.DataFrame:
    """Load all votes from the sheet (cached 5 min)."""
    try:
        ws = _get_votes_worksheet()
        if ws is None:
            return pd.DataFrame()
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _render_votes_stats():
    """Render comprehensive voting statistics dashboard."""
    df = _load_votes_cached()

    if df.empty:
        st.info("עדיין אין נתוני הצבעות. היה הראשון לבחור חלופה!")
        return

    from datetime import timedelta
    cutoff_30 = datetime.now() - timedelta(days=30)
    cutoff_7  = datetime.now() - timedelta(days=7)
    df30 = df[df["timestamp"] >= cutoff_30].copy()
    df7  = df[df["timestamp"] >= cutoff_7].copy()

    total_all = len(df)
    total_30  = len(df30)
    total_7   = len(df7)

    # ── Summary KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("סה״כ בחירות", f"{total_all:,}")
    c2.metric("30 יום אחרונים", f"{total_30:,}")
    c3.metric("7 ימים אחרונים", f"{total_7:,}")
    # unique sessions
    if "session_hash" in df30.columns:
        unique_users = df30["session_hash"].nunique()
        c4.metric("משתמשים ייחודיים (30י׳)", f"{unique_users:,}")

    if df30.empty:
        st.caption("אין הצבעות ב-30 הימים האחרונים עדיין.")
        return

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 חלופות", "🏢 מנהלים", "📋 מסלולים", "⏱️ טרנד"])

    bar_colors_alt = {
        "חלופה משוקללת": "#2563eb", "הכי מדויקת": "#16a34a",
        "שארפ מקסימלי": "#ea580c", "שירות מוביל": "#7c3aed"
    }

    # ── Tab 1: Alternatives ──
    with tab1:
        alt_counts = df30["alternative"].value_counts().reset_index()
        alt_counts.columns = ["חלופה", "הצבעות"]
        alt_counts["אחוז"] = (alt_counts["הצבעות"] / total_30 * 100).round(1)
        colors = [bar_colors_alt.get(a, "#64748b") for a in alt_counts["חלופה"]]
        fig = go.Figure(go.Bar(
            x=alt_counts["חלופה"], y=alt_counts["הצבעות"],
            marker_color=colors,
            text=alt_counts["אחוז"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(height=260, margin=dict(t=20,b=40,l=5,r=5),
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(title="בחירות", gridcolor="#F1F5F9"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        _safe_plotly(fig, key="stats_alts")
        st.dataframe(alt_counts.set_index("חלופה"), use_container_width=True)

    # ── Tab 2: Managers ──
    with tab2:
        if "managers" in df30.columns:
            all_mgrs_voted = []
            for cell in df30["managers"].dropna():
                for m in str(cell).replace("،","|").split("|"):
                    m = m.strip()
                    if m:
                        all_mgrs_voted.append(m)
            if all_mgrs_voted:
                mc = pd.Series(all_mgrs_voted).value_counts().head(10).reset_index()
                mc.columns = ["מנהל", "ספירה"]
                fig2 = go.Figure(go.Bar(
                    x=mc["מנהל"], y=mc["ספירה"],
                    marker_color="#3A7AFE",
                    text=mc["ספירה"], textposition="outside",
                ))
                fig2.update_layout(height=280, margin=dict(t=20,b=60,l=5,r=5),
                    xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                    yaxis=dict(title="ספירה", gridcolor="#F1F5F9"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                _safe_plotly(fig2, key="stats_mgrs")
                st.dataframe(mc.set_index("מנהל"), use_container_width=True)
            else:
                st.info("אין נתוני מנהלים בהצבעות")
        else:
            st.info("עמודת מנהלים חסרה בנתונים")

    # ── Tab 3: Tracks ──
    with tab3:
        if "tracks" in df30.columns:
            all_tracks_voted = []
            for cell in df30["tracks"].dropna():
                for t in str(cell).split("|"):
                    t = t.strip()
                    if t:
                        all_tracks_voted.append(t)
            if all_tracks_voted:
                tc = pd.Series(all_tracks_voted).value_counts().head(12).reset_index()
                tc.columns = ["מסלול", "ספירה"]
                fig3 = go.Figure(go.Bar(
                    x=tc["מסלול"], y=tc["ספירה"],
                    marker_color="#15A46E",
                    text=tc["ספירה"], textposition="outside",
                ))
                fig3.update_layout(height=300, margin=dict(t=20,b=80,l=5,r=5),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                    yaxis=dict(title="ספירה", gridcolor="#F1F5F9"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                _safe_plotly(fig3, key="stats_tracks")
                st.dataframe(tc.set_index("מסלול"), use_container_width=True)
            else:
                st.info("אין נתוני מסלולים בהצבעות")
        else:
            st.info("עמודת מסלולים חסרה בנתונים")

    # ── Tab 4: Trend (daily votes last 30 days) ──
    with tab4:
        df30["date"] = df30["timestamp"].dt.date
        daily = df30.groupby("date").size().reset_index(name="הצבעות")
        if not daily.empty:
            fig4 = go.Figure(go.Scatter(
                x=daily["date"], y=daily["הצבעות"],
                mode="lines+markers",
                line=dict(color="#3A7AFE", width=2),
                marker=dict(size=5, color="#1F3A5F"),
                fill="tozeroy", fillcolor="rgba(58,122,254,0.08)",
            ))
            fig4.update_layout(height=220, margin=dict(t=10,b=40,l=5,r=5),
                xaxis=dict(title="תאריך", tickfont=dict(size=10)),
                yaxis=dict(title="בחירות ביום", gridcolor="#F1F5F9"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            _safe_plotly(fig4, key="stats_trend")

    st.caption(f"נתונים מתעדכנים כל 5 דקות · עדכון אחרון: {datetime.now().strftime('%H:%M:%S')} · מציג 30 ימים אחרונים")




# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════

# ── Derive active df based on manager filter ──
all_managers = sorted(df_long["manager"].unique().tolist())
# Apply the same default-unchecked logic used by the UI checkboxes.
# When selected_managers is None (first load), derive the default selection here too,
# so the optimizer candidate dataset already excludes blocked managers even before
# the user has interacted with any checkbox.
_sel_m_state = st.session_state.get("selected_managers")
if _sel_m_state is None:
    sel_m = [m for m in all_managers if not _is_default_unchecked(m)]
else:
    sel_m = _sel_m_state
real_sel = [m for m in sel_m if m in all_managers and m != "__none__"]
if real_sel and len(real_sel) < len(all_managers):
    df_active = df_long[df_long["manager"].isin(real_sel)].copy()
else:
    df_active = df_long

all_funds = sorted(df_active["fund"].unique().tolist())

# ── Derive results ──
res          = st.session_state.get("last_results")
recs         = _pick_recommendations(res.get("solutions_all") if res else None) if res else {}
baseline     = st.session_state.get("portfolio_baseline")
voting_conf  = hasattr(st, "secrets") and "gcp_service_account" in st.secrets
n_solutions  = len(res["solutions_all"]) if res and res.get("solutions_all") is not None else 0

# ── HEADER ──
render_header(len(df_active), len(all_managers), len(recs))

# ── PRODUCT TYPE SELECTOR ──
render_product_selector()

# ── QUICK FILTERS ──
render_quick_filters(df_active)

# ── FUND COMPARISON (above main layout, below quick filters) ──
render_fund_comparison(df_active, all_funds)

# ── MAIN 2-COL LAYOUT ──
col_left, col_right = st.columns([1.1, 0.9], gap="medium")

with col_left:
    run_clicked = render_mix_builder(df_active, all_funds)

with col_right:
    render_best_solution(recs, baseline)

# ── Handle calculation ──
if run_clicked:
    # Derive locked_pct
    locked_pct = None
    if st.session_state["locked_fund"] and st.session_state["total_amount"] > 0 and st.session_state["locked_amount"] > 0:
        locked_pct = round(st.session_state["locked_amount"] / st.session_state["total_amount"] * 100, 1)

    with st.spinner("⚡ מחשב... (חיפוש מואץ עם NumPy)"):
        try:
            sols, note = find_best_solutions(
                df=df_active,
                n_funds=st.session_state["n_funds"],
                step=st.session_state["step"],
                mix_policy=st.session_state["mix_policy"],
                include=st.session_state["include"],
                constraint=st.session_state["constraint"],
                targets=st.session_state["targets"],
                primary_rank=st.session_state["primary_rank"],
                locked_fund=st.session_state["locked_fund"],
                locked_weight_pct=locked_pct,
            )
            if sols is not None and not sols.empty:
                result = {
                    "solutions_all": sols,
                    "targets":       dict(st.session_state["targets"]),
                    "ts":            datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state["last_results"]  = result
                st.session_state["last_note"]     = note
                st.session_state["selected_alt"]  = None
                hist = st.session_state.get("run_history", [])
                hist.insert(0, result)
                st.session_state["run_history"] = hist[:3]
            else:
                st.warning(f"לא נמצאו תוצאות. {note}")
        except Exception as _e:
            st.error(f"שגיאה: {_e}")
    st.rerun()

# ── RESULTS (shown after calculation) ──
if res and recs:
    # Build rows list
    rows_list = []
    for _key, rrow, title in [
        ("weighted", recs.get("weighted"), "חלופה משוקללת"),
        ("accurate", recs.get("accurate"), "הכי מדויקת"),
        ("sharpe",   recs.get("sharpe"),   "שארפ מקסימלי"),
        ("service",  recs.get("service"),  "שירות מוביל"),
    ]:
        if rrow is None: continue
        r = dict(rrow)
        r["חלופה"]        = title
        r["weights_items"] = _weights_items(r.get("weights"), r.get("קופות",""), r.get("מסלולים",""), r.get("מנהלים_רשימה",""))
        r["משקלים"]       = _weights_short(r.get("weights"))
        rows_list.append(r)

    # Results strip
    render_results_strip(
        n_solutions=n_solutions,
        elapsed_note=st.session_state.get("last_note",""),
        qp_name=st.session_state.get("quick_profile_active")
    )

    # Results table
    render_results_table(rows_list, baseline, voting_conf)

    # Export
    top_df = pd.DataFrame(rows_list)
    exc, _ = st.columns([1, 6])
    with exc:
        st.download_button(
            "⬇️ ייצוא לאקסל",
            data=_export_excel(top_df, baseline),
            file_name="profit_mix_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_main"
        )

# ── HISTORY (expander) ──
render_history()

