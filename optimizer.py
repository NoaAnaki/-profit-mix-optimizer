from __future__ import annotations
# -*- coding: utf-8 -*-
"""
optimizer.py — Profit Mix Optimizer
מנוע האופטימיזציה: מחפש את שילובי הקרנות הטובים ביותר לפי יעדי חשיפה.
"""
import gc, itertools, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def _weights_for_n(n: int, step: int) -> np.ndarray:
    step = max(1, int(step))
    if n == 1:
        return np.array([[100]], dtype=float)
    if n == 2:
        ws = np.arange(0, 101, step)
        pairs = np.column_stack([ws, 100 - ws])
        return pairs.astype(float)
    out = []
    for w1 in range(0, 101, step):
        for w2 in range(0, 101 - w1, step):
            w3 = 100 - w1 - w2
            if w3 >= 0 and w3 % step == 0:
                out.append([w1, w2, w3])
    return np.array(out, dtype=float) if out else np.empty((0, 3), dtype=float)

def _prefilter_candidates(df, include, targets, cap, locked_fund):
    keys = [k for k, v in include.items() if v and k in ["foreign", "stocks", "fx", "illiquid"]]
    if not keys:
        keys = ["foreign", "stocks"]
    tmp = df.copy()
    score = np.zeros(len(tmp), dtype=float)
    for k in keys:
        score += np.abs(tmp[k].fillna(50.0).to_numpy() - float(targets.get(k, 0.0))) / 100.0
    tmp["_s"] = score
    if locked_fund:
        locked_mask = tmp["fund"].str.strip() == locked_fund.strip()
        locked_df = tmp[locked_mask]
        rest_df   = tmp[~locked_mask].sort_values("_s").head(max(cap - len(locked_df), 1))
        tmp = pd.concat([locked_df, rest_df])
    else:
        tmp = tmp.sort_values("_s").head(cap)
    return tmp.drop(columns=["_s"]).reset_index(drop=True)

def _hard_ok_vec(values, target, mode):
    if mode == "בדיוק":
        return np.abs(values - target) < 0.5
    if mode == "לפחות":
        return values >= target - 0.5
    if mode == "לכל היותר":
        return values <= target + 0.5
    return np.ones(len(values), dtype=bool)

def find_best_solutions(
    df, n_funds, step, mix_policy, include, constraint, targets, primary_rank,
    locked_fund="", locked_weight_pct: Optional[float] = None,
    max_solutions_scan=20000,
) -> Tuple[pd.DataFrame, str]:
    import gc
    targets = {k: float(v) for k, v in targets.items()}

    cap = 50 if n_funds == 2 else 35 if n_funds == 3 else 80
    df_scan = _prefilter_candidates(df, include, targets, cap=cap, locked_fund=locked_fund)

    weights_arr  = _weights_for_n(n_funds, step)
    if len(weights_arr) == 0:
        return pd.DataFrame(), "לא נמצאו שילובי משקלים. נסה צעד קטן יותר."
    weights_norm = weights_arr / 100.0

    metric_keys = ["foreign", "stocks", "fx", "illiquid"]
    active_soft = [k for k in metric_keys if include.get(k, False)] or ["foreign", "stocks"]
    soft_idx    = {k: i for i, k in enumerate(metric_keys)}
    hard_keys   = [(k, constraint[k][1]) for k in metric_keys
                   if constraint.get(k, ("רך", ""))[0] == "קשיח"]

    A       = df_scan[["foreign","stocks","fx","illiquid","sharpe","service"]].to_numpy(dtype=float)
    records = df_scan.reset_index(drop=True)

    locked_idx: Optional[int] = None
    if locked_fund:
        matches = records.index[records["fund"].str.strip() == locked_fund.strip()].tolist()
        if matches:
            locked_idx = matches[0]

    # ── פיצ׳ר 3: סינון משקלים לפי locked_weight_pct ──
    if locked_idx is not None and locked_weight_pct is not None:
        tol = max(step * 0.5, 0.5)
        # עמודה של הקרן הנעולה היא העמודה שמתאימה ל-locked_idx בקומבינציה
        # נסנן אחרי בחירת קומבינציה — שמור רק weights שבהם המשקל ב-locked_idx == locked_weight_pct
        # בשלב הלולאה נסנן ידנית
        pass  # handled in loop below

    if mix_policy == "אותו מנהל בלבד":
        groups = list(records.groupby("manager").groups.values())
        combo_source = itertools.chain.from_iterable(
            itertools.combinations(list(g), n_funds) for g in groups if len(g) >= n_funds
        )
    else:
        combo_source = itertools.combinations(range(len(records)), n_funds)

    solutions = []
    scanned   = 0
    MAX_STORED = 60000

    for combo in combo_source:
        if locked_idx is not None and locked_idx not in combo:
            continue
        scanned += 1
        if scanned > max_solutions_scan:
            break

        arr     = A[list(combo), :]
        w_arr   = weights_arr.copy()

        # ── פיצ׳ר 3: אם יש locked_weight_pct, סנן משקלים ──
        if locked_idx is not None and locked_weight_pct is not None:
            pos_in_combo = list(combo).index(locked_idx)
            # Snap to nearest weight step to guarantee a match
            snapped = round(locked_weight_pct / step) * step
            snapped = max(step, min(100 - step * (n_funds - 1), snapped))  # keep combo valid
            tol = step * 0.5 + 0.1
            mask_w = np.abs(w_arr[:, pos_in_combo] - snapped) <= tol
            w_arr = w_arr[mask_w]
            if len(w_arr) == 0:
                continue

        w_norm = w_arr / 100.0
        mix_all = np.einsum("wn,nm->wm", w_norm, np.nan_to_num(arr, nan=0.0))

        mask = np.ones(len(w_norm), dtype=bool)
        for k, mode in hard_keys:
            mask &= _hard_ok_vec(mix_all[:, soft_idx[k]], targets.get(k, 0.0), mode)
        if not mask.any():
            continue

        mix_ok    = mix_all[mask]
        w_ok      = w_arr[mask]
        score_arr = np.zeros(len(mix_ok))
        for k in active_soft:
            score_arr += np.abs(mix_ok[:, soft_idx[k]] - targets.get(k, 0.0)) / 100.0

        fund_labels  = [records.loc[i, "fund"]    for i in combo]
        track_labels = [records.loc[i, "track"]   for i in combo]
        managers     = [records.loc[i, "manager"] for i in combo]
        manager_set  = " | ".join(sorted(set(managers)))
        managers_per_fund = " | ".join(managers)  # ordered, one per fund

        # ── Feature 5: Sharpe validity check ──
        sharpe_vals = arr[:, 4]  # sharpe column
        sharpe_incomplete = bool(np.any(np.isnan(sharpe_vals) | (sharpe_vals == 0)))

        for wi in range(len(mix_ok)):
            solutions.append({
                "combo":             combo,
                "weights":           tuple(int(round(x)) for x in w_ok[wi]),
                "מנהלים":            manager_set,
                "מנהלים_רשימה":      managers_per_fund,
                "מסלולים":           " | ".join(track_labels),
                "קופות":             " | ".join(fund_labels),
                'חו"ל (%)'  :        float(mix_ok[wi, 0]),
                "ישראל (%)"  :        float(100.0 - mix_ok[wi, 0]),
                "מניות (%)"  :        float(mix_ok[wi, 1]),
                'מט"ח (%)'  :        float(mix_ok[wi, 2]),
                "לא־סחיר (%)" :       float(mix_ok[wi, 3]),
                "שארפ משוקלל":        np.nan if sharpe_incomplete else float(mix_ok[wi, 4]),
                "sharpe_incomplete":  sharpe_incomplete,
                "שירות משוקלל":       float(mix_ok[wi, 5]),
                "score"       :       float(score_arr[wi]),
            })

        if len(solutions) >= MAX_STORED:
            solutions.sort(key=lambda r: (r["score"], -r["שארפ משוקלל"], -r["שירות משוקלל"]))
            solutions = solutions[:10000]
            gc.collect()

    if not solutions:
        return pd.DataFrame(), "לא נמצאו פתרונות. נסה לרכך מגבלות, להגדיל צעד, או לשנות יעדים."

    df_sol = pd.DataFrame(solutions)
    del solutions
    gc.collect()

    note = f"נסרקו {min(scanned, max_solutions_scan):,} קומבינציות מתוך {len(df_scan)} קופות מסוננות."

    if primary_rank == "דיוק":
        df_sol = df_sol.sort_values(["score", "שארפ משוקלל", "שירות משוקלל"], ascending=[True, False, False])
    elif primary_rank == "שארפ":
        df_sol = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    elif primary_rank == "שירות ואיכות":
        df_sol = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])

    return df_sol, note

def _pick_three_distinct(df_sol, primary_rank):
    if df_sol.empty:
        return df_sol

    def mgr(row): return str(row["מנהלים"]).strip()

    sorted_primary = df_sol.copy()
    sorted_sharpe  = df_sol.sort_values(["שארפ משוקלל",  "score"], ascending=[False, True])
    sorted_service = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])

    def best_from(df_sorted, exclude_managers):
        for _, r in df_sorted.iterrows():
            if mgr(r) not in exclude_managers:
                return r
        return df_sorted.iloc[0]

    pick1 = best_from(sorted_primary, set())
    pick2 = best_from(sorted_sharpe,  set())
    pick3 = best_from(sorted_service, set())

    used_after_1 = {mgr(pick1)}
    if mgr(pick2) in used_after_1:
        pick2 = best_from(sorted_sharpe, used_after_1)

    used_after_2 = used_after_1 | {mgr(pick2)}
    if mgr(pick3) in used_after_2:
        pick3 = best_from(sorted_service, used_after_2)

    labels     = ["חלופה 1 – דירוג ראשי", "חלופה 2 – שארפ", "חלופה 3 – שירות ואיכות"]
    criterions = ["דיוק", "שארפ", "שירות ואיכות"]
    base = pick1.to_dict()
    rows = []
    for i, r in enumerate([pick1, pick2, pick3]):
        row = r.to_dict()
        row["חלופה"]        = labels[i]
        row["weights_items"] = _weights_items(row.get("weights"), row.get("קופות",""), row.get("מסלולים",""), row.get("מנהלים_רשימה",""))
        row["משקלים"]       = _weights_short(row.get("weights"))
        row["יתרון"]        = _make_advantage(criterions[i], row, base if i > 0 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def _weights_items(weights, funds_str, tracks_str, managers_str=""):
    try:    ws = list(weights)
    except: ws = []
    funds    = [s.strip() for s in (funds_str    or "").split("|") if s.strip()]
    tracks   = [s.strip() for s in (tracks_str   or "").split("|") if s.strip()]
    managers = [s.strip() for s in (managers_str or "").split("|") if s.strip()]
    n = max(len(ws), len(funds))
    return [
        {
            "pct":     f"{int(round(float(ws[i])))}%" if i < len(ws) else "?",
            "fund":    funds[i]    if i < len(funds)    else "",
            "track":   tracks[i]  if i < len(tracks)   else "",
            "manager": managers[i] if i < len(managers) else "",
        }
        for i in range(n)
    ]

def _weights_short(weights):
    if weights is None: return ""
    try:    w = [float(x) for x in weights]
    except: return ""
    return " / ".join(f"{int(round(x))}%" for x in w)

def _make_advantage(primary, row, base=None):
    score = row.get("score", 0)
    if primary == "דיוק":
        return f"מדויק ביותר ליעד (סטייה {score:.4f})"
    if primary == "שארפ":
        sh = float(row.get("שארפ משוקלל", 0) or 0)
        delta = sh - float((base or {}).get("שארפ משוקלל", sh) or sh)
        return f"שארפ {sh:.2f} (+{delta:.2f} מחלופה 1)"
    sv = float(row.get("שירות משוקלל", 0) or 0)
    delta = sv - float((base or {}).get("שירות משוקלל", sv) or sv)
    return f"שירות ואיכות {sv:.1f} (+{delta:.1f} מחלופה 1)"


# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────


# ── Results helpers ─────────────────────────────────────────────────────────
def _normalize_series(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def _pick_recommendations(df_sol_head):
    if df_sol_head is None or df_sol_head.empty:
        return {}
    df = df_sol_head.copy()
    if "score" not in df.columns:
        return {}

    accurate = df.loc[df["score"].idxmin()].to_dict()

    # Only use sharpe/service columns if they exist AND have at least one non-NaN value
    has_sharpe  = "שארפ משוקלל"  in df.columns and df["שארפ משוקלל"].notna().any()
    has_service = "שירות משוקלל" in df.columns and df["שירות משוקלל"].notna().any()

    best_sh      = df.loc[df["שארפ משוקלל"].idxmax()].to_dict()  if has_sharpe  else accurate
    best_service = df.loc[df["שירות משוקלל"].idxmax()].to_dict() if has_service else accurate

    score_n   = _normalize_series(df["score"])
    acc_n     = 1.0 - score_n
    sharpe_n  = _normalize_series(df["שארפ משוקלל"])  if has_sharpe  else pd.Series([0.0] * len(df), index=df.index)
    service_n = _normalize_series(df["שירות משוקלל"]) if has_service else pd.Series([0.0] * len(df), index=df.index)
    df["_weighted_pref"] = 0.45 * acc_n + 0.15 * sharpe_n + 0.40 * service_n
    weighted = df.loc[df["_weighted_pref"].idxmax()].to_dict()

    return {"weighted": weighted, "accurate": accurate, "sharpe": best_sh, "service": best_service}

def _weighted_metric_from_items(items, df_long, metric_key):
    """Weighted average of metric_key for a list of fund items.
    Match: 1) exact fund name  2) fallback manager+track.
    Returns NaN if no values found."""
    if not items or df_long is None or df_long.empty:
        return np.nan
    total_w = 0.0
    total_v = 0.0
    for it in items:
        try:
            pct = float(str(it.get("pct", "0")).replace("%", "") or 0)
        except Exception:
            continue
        if pct <= 0:
            continue
        fund    = str(it.get("fund",    "")).strip()
        track   = str(it.get("track",   "")).strip()
        manager = str(it.get("manager", "")).strip()
        match = df_long[df_long["fund"].str.strip() == fund] if fund else pd.DataFrame()
        if match.empty and manager and track:
            match = df_long[
                (df_long["manager"].str.strip() == manager) &
                (df_long["track"].str.strip()   == track)
            ]
        if match.empty:
            continue
        val = _to_float(match.iloc[0].get(metric_key, np.nan))
        if math.isnan(val):
            continue
        total_w += pct
        total_v += pct * val
    return (total_v / total_w) if total_w > 0 else np.nan

def _compute_weighted_returns_for_items(items, df_long):
    """Return dict of all weighted return metrics for an alternative's items."""
    return {
        "ret_ytd": _weighted_metric_from_items(items, df_long, "ret_ytd"),
        "ret_12m": _weighted_metric_from_items(items, df_long, "ret_12m"),
        "ret_36m": _weighted_metric_from_items(items, df_long, "ret_36m"),
        "ret_60m": _weighted_metric_from_items(items, df_long, "ret_60m"),
    }

def _manager_weights_from_items(items, manager_names):
    if not items: return []
    names = sorted([m for m in manager_names if isinstance(m, str) and m.strip()], key=len, reverse=True)
    agg = {}
    for it in items:
        fund = str(it.get("fund",""))
        pct  = float(str(it.get("pct","0")).replace("%","") or 0)
        chosen = None
        for n in names:
            if fund.strip().startswith(n) or (n in fund.strip()):
                chosen = n
                break
        if chosen is None: chosen = "אחר"
        agg[chosen] = agg.get(chosen, 0.0) + pct
    return sorted(agg.items(), key=lambda x: -x[1])

def _change_type_badge(cur_mgrs, prop_mgrs):
    cur_set  = set(m.strip() for m in cur_mgrs  if m.strip())
    prop_set = set(m.strip() for m in prop_mgrs if m.strip())
    if not cur_set:
        return ""
    if cur_set == prop_set:
        return "<span class='change-badge change-low'>✅ שינוי מסלול – אותו מנהל</span>"
    elif cur_set & prop_set:
        return "<span class='change-badge change-med'>⚠️ ניוד חלקי</span>"
    else:
        return "<span class='change-badge change-high'>🔴 ניוד מלא</span>"

# ─────────────────────────────────────────────
# AI explanation (Claude API)
