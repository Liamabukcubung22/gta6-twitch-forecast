#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case (/input/<id>.in, schema "makefun/twitch-ccu-case/v1") and
writes /output/<id>.out: for each of hours 2..24 of the release day (UTC),
five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's mean concurrent
viewers in the game's Twitch category.

Model
-----
The single most informative number the case gives us is hour 1's print:
how many people were already watching at 00:00-01:00 UTC, when only New
Zealand, Australia and (just barely) Europe had unlocked. Past big
launches show one clean regularity: the day's shape depends on that first
hour. A launch whose category is already huge at hour 1 (hundreds of
thousands of viewers) stays roughly flat -- a mid-morning dip, a modest
evening peak about 20-30% above hour 1. A launch whose category is still
nearly empty at hour 1 (a few hundred viewers, unlock still hours away)
explodes during the day, growing 10x-1000x by the evening. In log terms
the relation between log10(hour-1 viewers) and each hour's log-ratio
ln(viewers_h / viewers_1) is close to linear.

So for each hour h = 2..24 we fit, on the case's own "comparables" (past
big launches' release days, shipped inside the case file), a locally
weighted linear regression of ln(v_h / v_1) on log10(v_1), with a Gaussian
kernel (bandwidth 0.6 dex) centred on this case's log10(hour-1 print).
The fitted value is the median log-ratio; the spread is the weighted
residual RMS of the same fit, scaled by 0.85 (the residuals are
heavy-tailed -- a few launches behave wildly -- so the raw RMS overstates
the typical miss), and the five quantiles are median + z_tau * spread
with the standard-normal quantiles. Everything is computed from the case
file at run time; nothing is fetched, nothing is random.

Fallbacks: an hour with too few comparable curves (< 8, or kernel weight
< 3) falls back to the flat house shape for that hour; if the case has
no usable comparables at all, the whole forecast is the flat shape.
Hour 1's print is clamped to at least 1 viewer, and the regression's
input log10(hour-1) is clamped to the range the comparables actually
span: a launch bigger than anything on record is forecast like the
biggest launch on record, not by extrapolating the fitted line past
it (a backtest that fits on launches under 200k hour-1 viewers and
forecasts the five biggest launches scores 0.154 clamped versus 0.199
unclamped, house baseline 0.220).

Standard library only. Deterministic: the same case always yields the
same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at the five levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

BANDWIDTH = 0.6        # Gaussian kernel bandwidth, in dex of log10(hour 1)
SPREAD_SCALE = 0.85    # shrink the residual RMS (heavy-tailed residuals)
MIN_CURVES = 8         # curves needed to fit one hour
MIN_WEIGHT = 3.0       # kernel weight needed to fit one hour
SIGMA_FLOOR = 0.10     # never claim to be surer than this (log units)
SIGMA_CAP = 1.50       # never spread wider than this (log units)
MAX_VIEWERS = 1_000_000_000.0


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def release_day_curve(comp):
    """Hourly viewers {hour: viewers} for a comparable's release day.

    comp["day_start"] is 00:00 UTC of the release date; hour h covers
    [day_start + (h-1)h, day_start + h*h). Points with no viewers are
    skipped; a curve without an hour-1 point is useless and returns
    an empty mapping.
    """
    base = parse_ts(comp.get("day_start"))
    if base is None:
        return {}
    curve = {}
    for pt in comp.get("hourly") or []:
        t = parse_ts(pt.get("from"))
        v = pt.get("viewers")
        if t is None or v is None:
            continue
        delta_h = (t - base).total_seconds() / 3600.0
        if 0.0 <= delta_h < 24.0:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v >= 0:
                curve[int(delta_h) + 1] = v
    if 1 not in curve:
        return {}
    return curve


def fit_hour(curves, hour, target_logh1):
    """Locally weighted linear fit of ln(v_h / v_1) on log10(v_1).

    Returns (median_log_ratio, sigma) at target_logh1, or None when the
    hour has too little data to fit.
    """
    xs, ys, ws = [], [], []
    for curve in curves:
        p1 = curve.get(1)
        vh = curve.get(hour)
        if p1 is None or vh is None or p1 < 1 or vh < 1:
            continue
        lx = math.log10(p1)
        w = math.exp(-0.5 * ((lx - target_logh1) / BANDWIDTH) ** 2)
        if w < 1e-6:
            continue
        xs.append(lx)
        ys.append(math.log(vh / p1))
        ws.append(w)
    if len(xs) < MIN_CURVES or sum(ws) < MIN_WEIGHT:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(sw - 2.0, 1.0)
    sigma = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / dof)
    sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
    median = intercept + slope * target_logh1
    return median, sigma


def flat_quantiles(p1, hour):
    """Fallback shape: flat at hour 1 with a spread widening over the day
    (the house baseline's shape)."""
    p = max(float(p1), 1.0)
    spread = 0.15 * math.sqrt(hour - 1)
    return [p * math.exp(z * spread) for z in Z]


def forecast_case(case):
    """Build the {"hours": [...]} forecast object for one case dict."""
    known = case.get("known") or []
    p1 = 1.0
    if known:
        try:
            p1 = float(known[0].get("viewers") or 0) or 1.0
        except (TypeError, ValueError):
            p1 = 1.0
    p1 = max(p1, 1.0)

    curves = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve:
            curves.append(curve)

    target = math.log10(p1)
    if curves:
        lo = min(math.log10(c[1]) for c in curves if c.get(1))
        hi = max(math.log10(c[1]) for c in curves if c.get(1))
        target = min(max(target, lo), hi)
    rows = []
    for hour in range(2, 25):
        fit = fit_hour(curves, hour, target) if curves else None
        if fit is None:
            vals = flat_quantiles(p1, hour)
        else:
            median, sigma = fit
            vals = [p1 * math.exp(median + z * sigma * SPREAD_SCALE)
                    for z in Z]
        vals = [min(max(v, 1.0), MAX_VIEWERS) for v in vals]
        vals = sorted(round(v, 2) for v in vals)
        rows.append({"hour": hour, "q": vals})
    return {"hours": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError as exc:
        print("cannot list input dir %s: %s" % (in_dir, exc), file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
