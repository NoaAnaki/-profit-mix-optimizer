# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/charts.py
──────────────────────────────────────────
Plotly chart builders.  All charts use real datetime X-axes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Design tokens ─────────────────────────────────────────────────────────────

_PALETTE = [
    "#3A7AFE", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
    "#F97316", "#6366F1", "#14B8A6", "#FB7185",
]

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,249,252,1)",
    font=dict(family="Segoe UI, -apple-system, sans-serif", size=12, color="#374151"),
    margin=dict(l=10, r=10, t=45, b=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.35,
        xanchor="center", x=0.5, font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    hovermode="x unified",
)

def _base(fig: go.Figure, title: str = "", height: int = 430) -> go.Figure:
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#1F3A5F"), x=0.5),
        height=height,
    )
    fig.update_xaxes(type="date", tickformat="%b %Y", gridcolor="#E5E7EB")
    fig.update_yaxes(ticksuffix="%", gridcolor="#E5E7EB", zeroline=False)
    return fig


# ── 1. Time-series ────────────────────────────────────────────────────────────

def build_timeseries(df: pd.DataFrame, title: str = "חשיפה לאורך זמן",
                     height: int = 460) -> go.Figure:
    """
    One line per (manager × track × allocation_name).
    Yearly points shown as larger markers with dashed lines.
    Monthly points shown as solid lines.
    """
    fig = go.Figure()

    groups = df.groupby(["manager", "track", "allocation_name"])
    for idx, (key, gdf) in enumerate(groups):
        manager, track, alloc = key
        label = f"{manager} {track} — {alloc}"
        color = _PALETTE[idx % len(_PALETTE)]
        gdf = gdf.sort_values("date")

        # Separate yearly / monthly for styling
        gdf_m = gdf[gdf.get("frequency", pd.Series(["monthly"] * len(gdf))) == "monthly"] \
            if "frequency" in gdf.columns else gdf
        gdf_y = gdf[gdf.get("frequency", pd.Series(["monthly"] * len(gdf))) == "yearly"] \
            if "frequency" in gdf.columns else pd.DataFrame()

        # Monthly → solid line
        if not gdf_m.empty:
            fig.add_trace(go.Scatter(
                x=gdf_m["date"], y=gdf_m["allocation_value"],
                mode="lines", name=label, line=dict(color=color, width=2.2),
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>",
            ))

        # Yearly → dashed line + markers (only if not already covered by monthly)
        if not gdf_y.empty:
            fig.add_trace(go.Scatter(
                x=gdf_y["date"], y=gdf_y["allocation_value"],
                mode="lines+markers",
                name=f"{label} (שנתי)", line=dict(color=color, width=1.5, dash="dot"),
                marker=dict(size=6, color=color),
                legendgroup=label, showlegend=gdf_m.empty,
                hovertemplate=f"<b>{label} (שנתי)</b><br>%{{x|%Y}}<br>%{{y:.2f}}%<extra></extra>",
            ))

    return _base(fig, title, height)


# ── 2. Snapshot bar ───────────────────────────────────────────────────────────

def build_snapshot(df: pd.DataFrame, snapshot_date: pd.Timestamp,
                   title: Optional[str] = None, height: int = 380) -> go.Figure:
    past = df[df["date"] <= snapshot_date]
    if past.empty:
        return go.Figure().update_layout(**_LAYOUT, title="אין נתונים לתאריך זה")

    idx = past.groupby(["manager", "track", "allocation_name"])["date"].idxmax()
    snap = past.loc[idx].copy()
    snap["label"] = snap["manager"] + " " + snap["track"] + " — " + snap["allocation_name"]
    snap = snap.sort_values("allocation_value", ascending=True)
    title = title or f"Snapshot — {snapshot_date.strftime('%b %Y')}"

    fig = go.Figure(go.Bar(
        x=snap["allocation_value"], y=snap["label"],
        orientation="h", marker_color=_PALETTE[0],
        text=snap["allocation_value"].map(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT, height=max(height, 50 + 28 * len(snap)),
        title=dict(text=title, font=dict(size=14, color="#1F3A5F"), x=0.5),
        xaxis=dict(ticksuffix="%", gridcolor="#E5E7EB"),
        yaxis=dict(tickfont=dict(size=10)), showlegend=False,
    )
    return fig


# ── 3. Delta chart ────────────────────────────────────────────────────────────

def build_delta(df: pd.DataFrame, date_a: pd.Timestamp,
                date_b: pd.Timestamp) -> tuple[go.Figure, pd.DataFrame]:

    def _snap(dt: pd.Timestamp) -> pd.DataFrame:
        p = df[df["date"] <= dt]
        if p.empty:
            return pd.DataFrame()
        i = p.groupby(["manager", "track", "allocation_name"])["date"].idxmax()
        return p.loc[i].copy()

    sa, sb = _snap(date_a), _snap(date_b)
    if sa.empty or sb.empty:
        return go.Figure().update_layout(**_LAYOUT, title="אין נתונים"), pd.DataFrame()

    merged = sa.merge(sb, on=["manager", "track", "allocation_name"], suffixes=("_a", "_b"))
    merged["delta"] = merged["allocation_value_b"] - merged["allocation_value_a"]
    merged["label"] = merged["manager"] + " " + merged["track"] + " — " + merged["allocation_name"]
    merged = merged.sort_values("delta")

    fig = go.Figure(go.Bar(
        x=merged["delta"], y=merged["label"], orientation="h",
        marker_color=["#EF4444" if d < 0 else "#10B981" for d in merged["delta"]],
        text=merged["delta"].map(lambda v: f"{v:+.1f}pp"), textposition="outside",
        customdata=merged[["allocation_value_a", "allocation_value_b"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{date_a.strftime('%b %Y')}: %{{customdata[0]:.1f}}%<br>"
            f"{date_b.strftime('%b %Y')}: %{{customdata[1]:.1f}}%<br>"
            "שינוי: %{x:+.1f} נ\"א<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_LAYOUT, height=max(380, 50 + 28 * len(merged)),
        title=dict(text=f"שינוי: {date_a.strftime('%b %Y')} → {date_b.strftime('%b %Y')}",
                   font=dict(size=14, color="#1F3A5F"), x=0.5),
        xaxis=dict(ticksuffix="pp", gridcolor="#E5E7EB",
                   zeroline=True, zerolinecolor="#9CA3AF"),
        yaxis=dict(tickfont=dict(size=10)), showlegend=False,
    )

    delta_df = merged[["manager", "track", "allocation_name",
                        "allocation_value_a", "allocation_value_b", "delta"]].rename(columns={
        "allocation_value_a": f"ערך ב-{date_a.strftime('%b %Y')}",
        "allocation_value_b": f"ערך ב-{date_b.strftime('%b %Y')}",
        "delta": "שינוי (pp)",
    })
    return fig, delta_df


# ── 4. Heatmap ────────────────────────────────────────────────────────────────

def build_heatmap(df: pd.DataFrame, title: str = "Heatmap — חשיפה חודשית",
                  height: int = 400) -> go.Figure:
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["row"] = d["manager"] + " " + d["track"] + " — " + d["allocation_name"]
    pivot = (d.groupby(["row", "ym"])["allocation_value"].mean()
              .unstack("ym"))
    pivot = pivot[sorted(pivot.columns)]
    col_lbl = [c.strftime("%b %Y") for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=col_lbl, y=pivot.index.tolist(),
        colorscale="Blues",
        text=np.round(pivot.values, 1), texttemplate="%{text:.1f}%",
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(ticksuffix="%", len=0.8),
    ))
    fig.update_layout(
        **_LAYOUT, height=max(height, 80 + 35 * len(pivot)),
        title=dict(text=title, font=dict(size=14, color="#1F3A5F"), x=0.5),
        xaxis=dict(type="category", tickfont=dict(size=10), tickangle=-45),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ── 5. Summary statistics ─────────────────────────────────────────────────────

def build_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, gdf in df.groupby(["manager", "track", "allocation_name"]):
        gdf = gdf.sort_values("date")
        vals = gdf["allocation_value"].dropna()
        if vals.empty:
            continue
        diffs = vals.diff().dropna()
        last_val = vals.iloc[-1]

        # 12-month change
        max_date = gdf["date"].max()
        yr_ago_df = gdf[gdf["date"] <= max_date - pd.DateOffset(months=12)]
        yr_ago = yr_ago_df.iloc[-1]["allocation_value"] if not yr_ago_df.empty else np.nan

        rows.append({
            "מנהל": key[0], "מסלול": key[1], "רכיב": key[2],
            "ממוצע (%)": round(vals.mean(), 2),
            "מינימום (%)": round(vals.min(), 2),
            "מקסימום (%)": round(vals.max(), 2),
            "סטיית תקן": round(vals.std(), 2),
            "שינוי חודשי ממוצע": round(diffs.mean(), 2) if not diffs.empty else np.nan,
            "שינוי חודשי מקס׳": round(diffs.abs().max(), 2) if not diffs.empty else np.nan,
            "שינוי 12 חודש (pp)": round(last_val - yr_ago, 2) if not np.isnan(yr_ago) else np.nan,
            "ערך אחרון (%)": round(last_val, 2),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── 6. Ranking chart ──────────────────────────────────────────────────────────

def build_ranking(df: pd.DataFrame, title: str = "דירוג מנהלים לאורך זמן",
                  height: int = 400) -> go.Figure:
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["row"] = d["manager"] + " " + d["track"]
    pivot = d.groupby(["row", "ym"])["allocation_value"].mean().unstack("ym")
    pivot = pivot[sorted(pivot.columns)]
    rank_df = pivot.rank(axis=0, ascending=False, method="min")

    fig = go.Figure()
    for idx, row in enumerate(rank_df.index):
        fig.add_trace(go.Scatter(
            x=rank_df.columns, y=rank_df.loc[row],
            mode="lines+markers", name=row,
            line=dict(color=_PALETTE[idx % len(_PALETTE)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{row}</b><br>%{{x|%b %Y}}<br>דירוג: %{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(
        **_LAYOUT, height=height,
        title=dict(text=title, font=dict(size=14, color="#1F3A5F"), x=0.5),
    )
    fig.update_xaxes(type="date", tickformat="%b %Y", gridcolor="#E5E7EB")
    fig.update_yaxes(autorange="reversed", title="דירוג (1=גבוה)", gridcolor="#E5E7EB", dtick=1)
    return fig
