"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at the case target, as five quantiles (0.05/0.25/0.5/0.75/0.95).

Method (standard library only, deterministic, no network):

* Exophase does not update continuously: a game's page sits frozen for
  hours, then refreshes -- players tracked jumps and every rate but the
  instant (100%) trophy falls, because the newly tracked players have
  earned almost nothing yet. The practice histories show exactly that
  shape: 638 players tracked and every rate identical for 3+ hours.
  The house's own notes say the first refresh took tracked players to
  3,868 and total trophies earned to 4,276, so the non-100% rates fell
  to roughly a fifth of their launch-night values on average (the sum
  of the rates fell from ~148% to ~111%, the 100% trophy included).
  A forecaster that just extends the last rate is therefore punished
  whenever a refresh lands inside its horizon, which is what the
  practice cases are built to test.
* Per trophy, work on the logit scale with x = ln(1 + hours since
  release). Use only distinct snapshots (consecutive identical reads
  carry no new information) and fit a recency-weighted least-squares
  line of logit(rate) on x. The slope is shrunk toward the
  cross-trophy median slope, so a trophy with little movement borrows
  strength from the rest of the list, and clamped so no single trophy
  runs away. Read the line at the target's x: log-time growth slows
  as the launch week goes on, which is also how dilution slows as
  the influx of new tracked players decays.
* When the history has (almost) never moved -- the practice shape --
  the trend is unidentified, so blend in a count-space refresh prior:
  the tracked-player count is expected to jump (a backlog factor
  that grows with the time since the last change plus the horizon),
  old earners carry over, and a newly tracked player earns a given
  trophy with only a small fraction (~0.07) of the old cohort's rate
  by the target -- except a trophy already at ~100%, which new
  players earn immediately, so it stays at ~100%.
* Spread (logit sigma) = base uncertainty + a random-walk term that
  grows with sqrt(horizon in days) + a fixed refresh component (large
  when no refresh has ever been seen, smaller once refreshes are in
  the history) + the trend fit's prediction error. It is floored and
  capped so a 90-minute horizon still carries a wide band (a refresh
  can land at any time) and a 4.5-day horizon is wider still, but
  never absurdly wide. Quantiles are inv-logit(median logit + z*sigma)
  with the standard-normal z for each level, rounded to 4 decimals
  half-up like the house baseline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread pieces, in logit units.
SIGMA_BASE = 0.30
SIGMA_RW_PER_SQRT_DAY = 0.5
SIGMA_REFRESH_UNSEEN = 0.65
SIGMA_REFRESH_SEEN = 0.30
SIGMA_FLOOR = 0.60
SIGMA_CAP = 1.50

# Refresh prior, used when the history has barely moved.
NEW_EARN_ALPHA = 0.07      # new joiner's earn prob, as a share of the old rate
REFRESH_BACKLOG = 5.0      # scale of the tracked-player jump at a refresh
REFRESH_TIMESCALE_H = 5.0  # hours over which the backlog builds


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO-8601 to an aware datetime; a bare date means midnight UTC."""
    if not s:
        return None
    s = s.strip()
    if len(s) == 10:
        s += "T00:00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def distinct_snapshots(case):
    """History collapsed to reads where rates or players actually changed.

    Returns a list of (hours_since_release, players, rates) with
    consecutive duplicates removed; the flat stretches between
    refreshes carry no signal about the trend.
    """
    release = parse_time(case.get("game", {}).get("release_date"))
    snaps = []
    prev_key = None
    for read in case.get("history", []):
        at = parse_time(read.get("at"))
        if at is None:
            continue
        hours = (at - release).total_seconds() / 3600.0 if release else 0.0
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append((hours, read.get("players_tracked"), rates))
    return snaps


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / max(1, len(xs)))
    return (a, b, math.sqrt(max(0.0, var)))


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_h = case["horizon_minutes"] / 60.0

    release = parse_time(case.get("game", {}).get("release_date"))
    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if release and target:
        target_h = (target - release).total_seconds() / 3600.0
    else:
        target_h = horizon_h
    x_target = math.log1p(max(0.0, target_h))

    snaps = distinct_snapshots(case)
    n_distinct = len(snaps)
    last = history[-1]
    last_rates = last.get("rates") or []
    last_n = last.get("players_tracked") or 0

    # Time since the last actual change in the page (stale time): a long
    # flat stretch means a refresh is overdue and its backlog is large.
    if snaps:
        stale_h = max(0.0, (at - parse_time(history[-1].get("at"))).total_seconds() / 3600.0) \
            if at and history[-1].get("at") else 0.0
        # Recompute stale from the last distinct snapshot's clock: the
        # last read's time is close enough, but use the snapshot hours.
        stale_h = max(0.0, target_h - horizon_h - snaps[-1][0])
    else:
        stale_h = horizon_h

    # ---- first pass: per-trophy trend fits, to get the median slope ----
    fits = []
    for i in range(n):
        xs, ys, ws = [], [], []
        for (h, _np, rates) in snaps:
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            xs.append(math.log1p(max(0.0, h)))
            ys.append(logit(r))
            # Recency weight: halflife ~36h from the last snapshot.
            age_h = max(0.0, snaps[-1][0] - h)
            ws.append(0.5 ** (age_h / 36.0))
        if len(xs) >= 3:
            fit = weighted_line(xs, ys, ws)
            if fit is not None:
                a, b, resid = fit
                b = max(-1.2, min(2.5, b))
                fits.append((a, b, resid, xs, ys, ws))
            else:
                fits.append(None)
        else:
            fits.append(None)
    slopes = [f[1] for f in fits if f is not None]
    median_slope = median(slopes) if slopes else 0.0

    # ---- refresh prior (count space) ----
    # Expected tracked-player multiplier at the next refresh. Once the
    # history shows refreshes, use their observed jump sizes (the
    # median N ratio between distinct snapshots); with no refresh ever
    # seen, use the backlog prior, which grows with the time since the
    # last change plus the horizon.
    jump_ratios = []
    for j in range(1, len(snaps)):
        n_prev, n_now = snaps[j - 1][1] or 0, snaps[j][1] or 0
        if n_prev and n_now and n_now > n_prev:
            jump_ratios.append(n_now / n_prev)
    if jump_ratios:
        n_mult = median(jump_ratios)
    else:
        backlog = 1.0 - math.exp(-(stale_h + horizon_h) / REFRESH_TIMESCALE_H)
        n_mult = 1.0 + REFRESH_BACKLOG * backlog
    refresh_seen = n_distinct >= 3

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            # Fall back to the most recent non-null read for this trophy.
            r_last = 1.0
            for read in reversed(history):
                rates = read.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    r_last = rates[i]
                    break
        r_last = clamp_pct(r_last)

        # Trend median.
        fit = fits[i]
        if fit is not None:
            a, b, resid, xs, ys, ws = fit
            # Shrink the slope toward the cross-trophy median: more
            # shrinkage when the trophy has few distinct points. Refit
            # the intercept at the data centroid so shrinking the slope
            # does not shift the line away from the observed rates.
            shrink = len(xs) / (len(xs) + 2.0)
            b_s = median_slope + shrink * (b - median_slope)
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / sw
            my = sum(w * y for w, y in zip(ws, ys)) / sw
            a_s = my - b_s * mx
            trend_logit = a_s + b_s * x_target
            # Prediction error of the fit at the target (leverage).
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / sw if sw else 0.0
            sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
            lev = 1.0 + ((x_target - mx) ** 2 / sxx if sxx > 1e-12 else 1.0)
            sigma_fit = min(0.8, resid * math.sqrt(max(0.0, lev)))
        else:
            trend_logit = logit(r_last)
            sigma_fit = 0.0

        # Refresh prior median (count space): old earners carry over,
        # new joiners add a small share, denominator jumps by n_mult.
        if r_last >= 99.5:
            alpha = 1.0
        elif r_last >= 10.0:
            alpha = NEW_EARN_ALPHA + (1.0 - NEW_EARN_ALPHA) * (r_last - 10.0) / 90.0
        else:
            alpha = NEW_EARN_ALPHA
        prior_rate = r_last * (1.0 + alpha * (n_mult - 1.0)) / n_mult
        prior_logit = logit(prior_rate)

        # Blend: the trend earns its weight from distinct snapshots.
        # One distinct snapshot is no trend at all (its slope is
        # unidentified), so a flat history leans wholly on the
        # refresh prior; each further refresh earns the trend weight.
        w_trend = (n_distinct - 1.0) / (n_distinct + 1.0) if n_distinct >= 1 else 0.0
        med_logit = w_trend * trend_logit + (1.0 - w_trend) * prior_logit
        med_logit = max(logit(0.02), min(logit(99.5), med_logit))
        # A trophy already at ~100% is earned on starting the game,
        # so newly tracked players earn it too: it stays at ~100%,
        # and a wide band around it would only pay pinball for nothing.
        if r_last >= 99.5:
            med_logit = logit(99.99)

        # Spread.
        sigma_refresh = SIGMA_REFRESH_SEEN if refresh_seen else SIGMA_REFRESH_UNSEEN
        sigma = math.sqrt(
            SIGMA_BASE ** 2
            + (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + sigma_refresh ** 2
            + sigma_fit ** 2
        )
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        if r_last >= 99.5:
            sigma = min(sigma, 0.25)

        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(med_logit + z * sigma))
            q = min(100.0, max(0.01, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
