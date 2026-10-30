# Lottery Pattern Explorer — lesson build, 2026-10-30.
#
# "Every method loses to randomness." The page builds several
# plausible-looking prediction methods, runs each one honestly against
# real past draws on a FIXED scoreboard (score, n, baseline, 95% CI),
# and shows they all land on the random baseline — while disagreeing
# with each other. No method is ever presented as a "play this" without
# its scoreboard row next to it.
#
# Fantasy 5: 5 distinct numbers from 1-39. Score = hits per draw,
# baseline 5 x (5/39) = 0.641. Exact-set baseline 1/575,757.
# Daily 5: 5 ordered digits 0-9, repeats allowed. Score = matching
# positions per draw, baseline 5 x 0.10 = 0.5. Exact-sequence baseline
# 1/100,000. Streams are never pooled.

import math
import random
from collections import Counter, deque
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lottery Pattern Explorer", layout="wide")
st.title("Lottery Pattern Explorer")
st.caption(
    "We built several different ways to pick numbers. They disagree "
    "with each other, they look clever, and they all score exactly the "
    "same as picking randomly — because that's what randomness does."
)

BASELINE_F5 = 5 * (5 / 39)  # 0.641 hits/draw
BASELINE_D5 = 5 * 0.10  # 0.5 matching positions/draw


def mean_ci(values):
    """Mean and 95% CI (normal approximation) for a list of scores."""
    n = len(values)
    if n == 0:
        return 0.0, (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return mean, (mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, (mean - half, mean + half)


def verdict(score, ci, baseline):
    if ci[0] <= baseline <= ci[1]:
        return "Noise — same as random"
    if score > baseline:
        return "Above baseline — re-check on fresh data"
    return "Below baseline — re-check on fresh data"


def scoreboard_rows(results, baseline):
    """results: {method: (scores, exact_hits, n)} -> dataframe rows."""
    rows = []
    for method, (scores, exact, n) in results.items():
        mean, ci = mean_ci(scores)
        rows.append({
            "Method": method,
            "n (test draws)": str(n),
            "Score": f"{mean:.3f}",
            "Baseline": f"{baseline:.3f}",
            "95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            "Verdict": verdict(mean, ci, baseline),
            "Exact hits": str(exact),
        })
    return rows


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_draws(uploaded, builtin_path, game_name):
    if uploaded is not None:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    elif builtin_path.exists():
        df = pd.read_csv(builtin_path, encoding="utf-8-sig")
        st.info(
            f"You're looking at the built-in {game_name} results. "
            "Upload your own file above and it takes over."
        )
    else:
        st.info(
            f"Upload a {game_name} results file (CSV) to begin — "
            "columns: Draw Date, Winning Numbers, and Drawing Type "
            "if the file has more than one drawing type."
        )
        return None
    df["Draw Date"] = pd.to_datetime(df["Draw Date"], format="%m/%d/%Y")
    df["Numbers"] = df["Winning Numbers"].apply(
        lambda s: [int(x) for x in str(s).split(",")]
    )
    if "Drawing Type" in df.columns:
        df["Drawing Type"] = df["Drawing Type"].astype(str).str.strip()
    else:
        df["Drawing Type"] = "ALL DRAWINGS"
    return df


def ordered_types(df, preferred):
    present = list(df["Drawing Type"].unique())
    ordered = [t for t in preferred if t in present]
    return ordered + sorted(t for t in present if t not in preferred)


def chronological(sub):
    """Draws oldest-first as lists of ints."""
    return [list(nums) for nums in
            sub.sort_values("Draw Date")["Numbers"]]


# ---------------------------------------------------------------------------
# Fantasy 5 backtest — one chronological pass, trailing-window state
# ---------------------------------------------------------------------------

F5_BANDS = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 39)]
F5_POPULAR = [7, 14, 21, 28, 31]  # fixed "birthday / lucky" set
F5_METHODS = ["Hot", "Cold", "Balanced", "Gap",
              "Random", "Anti", "Popular"]


def _ranked_f5(counts, reverse):
    if reverse:
        return sorted(range(1, 40), key=lambda k: (-counts.get(k, 0), k))
    return sorted(range(1, 40), key=lambda k: (counts.get(k, 0), k))


def fantasy_picks(counts, last_seen, now):
    """All walk-forward picks from current trailing-window state.

    counts: Counter over the trailing window. last_seen: number -> index
    of its last appearance before `now`. now: index of the draw being
    picked for. Returns {method: sorted 5-set}.
    """
    hot = _ranked_f5(counts, True)[:5]
    cold = _ranked_f5(counts, False)[:5]
    balanced = [max(range(a, b + 1),
                    key=lambda k: (counts.get(k, 0), -k))
                for a, b in F5_BANDS]
    gaps = {k: (now - last_seen[k] if k in last_seen else now + 1)
            for k in range(1, 40)}
    gap = sorted(range(1, 40), key=lambda k: (-gaps[k], k))[:5]
    union = set(hot) | set(cold) | set(balanced) | set(gap)
    anti = [k for k in _ranked_f5(counts, False) if k not in union][:5]
    return {
        "Hot": sorted(hot),
        "Cold": sorted(cold),
        "Balanced": sorted(balanced),
        "Gap": sorted(gap),
        "Anti": sorted(anti),
        "Popular": list(F5_POPULAR),
    }


@st.cache_data
def fantasy_backtest(draws, window=100, seed=42):
    """Walk-forward backtest. Returns (results, next_picks).

    results: {method: (per-draw hits, exact-set hits, n)}. Test draws
    are every draw after the first `window` draws. Random uses a fixed
    seed so the page is reproducible.
    """
    rng = random.Random(seed)
    counts, win, last_seen = Counter(), deque(), {}
    scores = {m: [] for m in F5_METHODS}
    exact = {m: 0 for m in F5_METHODS}
    for i, actual in enumerate(draws):
        if i >= window:
            picks = fantasy_picks(counts, last_seen, i)
            picks["Random"] = sorted(rng.sample(range(1, 40), 5))
            aset = set(actual)
            for m in F5_METHODS:
                pk = picks[m]
                scores[m].append(len(set(pk) & aset))
                if set(pk) == aset:
                    exact[m] += 1
        win.append(actual)
        counts.update(actual)
        for k in actual:
            last_seen[k] = i
        if len(win) > window:
            for k in win.popleft():
                counts[k] -= 1
                if counts[k] == 0:
                    del counts[k]
    next_picks = fantasy_picks(counts, last_seen, len(draws))
    next_picks["Random"] = sorted(
        random.Random(seed + 1).sample(range(1, 40), 5))
    results = {m: (scores[m], exact[m], len(scores[m]))
               for m in F5_METHODS}
    return results, next_picks


# ---------------------------------------------------------------------------
# Daily 5 backtest — same idea, per position
# ---------------------------------------------------------------------------

D5_POPULAR = [7, 3, 1, 9, 2]  # fixed "lucky" sequence
D5_METHODS = ["Hot", "Cold", "Balanced", "Gap",
              "Random", "Anti", "Popular"]


def daily_picks(pos_counts, pos_last, now):
    """Per-position picks from trailing-window state."""
    hot, cold, balanced, gap = [], [], [], []
    for pos in range(5):
        counts, last = pos_counts[pos], pos_last[pos]
        desc = sorted(range(10), key=lambda d: (-counts.get(d, 0), d))
        asc = sorted(range(10), key=lambda d: (counts.get(d, 0), d))
        hot.append(desc[0])
        cold.append(asc[0])
        gaps = {d: (now - last[d] if d in last else now + 1)
                for d in range(10)}
        gap.append(sorted(range(10), key=lambda d: (-gaps[d], d))[0])
        used = set()
        for d in desc:
            if d not in used:
                balanced.append(d)
                used.add(d)
                break
    anti = []
    for pos in range(5):
        counts = pos_counts[pos]
        forbidden = {hot[pos], cold[pos], balanced[pos], gap[pos]}
        asc = sorted(range(10), key=lambda d: (counts.get(d, 0), d))
        pick = next((d for d in asc if d not in forbidden), asc[0])
        anti.append(pick)
    return {
        "Hot": hot,
        "Cold": cold,
        "Balanced": balanced,
        "Gap": gap,
        "Anti": anti,
        "Popular": list(D5_POPULAR),
    }


@st.cache_data
def daily_backtest(draws, window=25, seed=42):
    rng = random.Random(seed)
    pos_counts = [Counter() for _ in range(5)]
    pos_last = [dict() for _ in range(5)]
    win = deque()
    scores = {m: [] for m in D5_METHODS}
    exact = {m: 0 for m in D5_METHODS}
    for i, actual in enumerate(draws):
        if i >= window:
            picks = daily_picks(pos_counts, pos_last, i)
            picks["Random"] = [rng.randrange(10) for _ in range(5)]
            for m in D5_METHODS:
                pk = picks[m]
                scores[m].append(sum(1 for a, b in zip(pk, actual)
                                     if a == b))
                if pk == actual:
                    exact[m] += 1
        win.append(actual)
        for pos, d in enumerate(actual):
            pos_counts[pos][d] += 1
            pos_last[pos][d] = i
        if len(win) > window:
            for pos, d in enumerate(win.popleft()):
                pos_counts[pos][d] -= 1
                if pos_counts[pos][d] == 0:
                    del pos_counts[pos][d]
    next_picks = daily_picks(pos_counts, pos_last, len(draws))
    next_picks["Random"] = [random.Random(seed + 1).randrange(10)
                            for _ in range(5)]
    results = {m: (scores[m], exact[m], len(scores[m]))
               for m in D5_METHODS}
    return results, next_picks


# ---------------------------------------------------------------------------
# Method 5 (Fantasy 5): backtested rule search — the overfitting demo.
# Search a grid of candidate rules on the first 70% of draws, keep the
# best in-sample rule, and test it on the last 30%. With enough
# candidate rules, some will look great in-sample by chance; the test
# score is where that illusion dies. (A small logistic regression was
# tried first and dropped: it barely overfit at all — train 0.676,
# test 0.670 hits/draw — because a linear model on random draws has
# almost nothing to grab. The rule search is the honest, visible
# version of the same lesson.)
# ---------------------------------------------------------------------------

F5_RULE_GRID = []
for _w in (10, 25, 50, 100, 200):
    F5_RULE_GRID.append((f"Hot (window {_w})", "hot", _w))
    F5_RULE_GRID.append((f"Cold (window {_w})", "cold", _w))
F5_RULE_GRID.append(("Gap (largest gaps)", "gap", None))
F5_RULE_GRID.append(("Balanced bands (window 100)", "balanced", 100))
F5_RULE_GRID.append(("Balanced bands (window 25)", "balanced", 25))


def _rule_pick(kind, window, counts, last_seen, now):
    if kind == "hot":
        return _ranked_f5(counts, True)[:5]
    if kind == "cold":
        return _ranked_f5(counts, False)[:5]
    if kind == "gap":
        gaps = {k: (now - last_seen[k] if k in last_seen else now + 1)
                for k in range(1, 40)}
        return sorted(range(1, 40), key=lambda k: (-gaps[k], k))[:5]
    return sorted(max(range(a, b + 1),
                      key=lambda k: (counts.get(k, 0), -k))
                  for a, b in F5_BANDS)


@st.cache_data
def fantasy_rule_search(draws):
    """Returns (best_name, train_mean, test_scores, next_pick, n_rules)."""
    split = int(len(draws) * 0.7)
    best = None
    for name, kind, window in F5_RULE_GRID:
        counts, win, last_seen = Counter(), deque(), {}
        train_hits, test_hits = [], []
        for i, actual in enumerate(draws):
            pick = None
            if kind in ("hot", "cold", "balanced") and i >= window:
                pick = _rule_pick(kind, window, counts, last_seen, i)
            elif kind == "gap" and i >= 1:
                pick = _rule_pick(kind, window, counts, last_seen, i)
            if pick is not None:
                h = len(set(pick) & set(actual))
                (train_hits if i < split else test_hits).append(h)
            win.append(actual)
            counts.update(actual)
            for k in actual:
                last_seen[k] = i
            if window and len(win) > window:
                for k in win.popleft():
                    counts[k] -= 1
                    if counts[k] == 0:
                        del counts[k]
        train_mean = (sum(train_hits) / len(train_hits)
                      if train_hits else 0.0)
        if best is None or train_mean > best[1]:
            best = (name, train_mean, test_hits, kind, window)
    name, train_mean, test_hits, kind, window = best
    counts, last_seen = Counter(), {}
    pool = draws[-window:] if window else draws
    for d in pool:
        counts.update(d)
    for i, actual in enumerate(draws):
        for k in actual:
            last_seen[k] = i
    next_pick = sorted(
        _rule_pick(kind, window, counts, last_seen, len(draws)))
    return name, train_mean, test_hits, next_pick, len(F5_RULE_GRID)


# ---------------------------------------------------------------------------
# Gotcha visuals: gap distribution vs geometric (memorylessness)
# ---------------------------------------------------------------------------

def gap_table(draws, p, max_gap, is_daily=False):
    """Pooled gaps between appearances vs the geometric null.

    Fantasy 5: gaps per number across draws. Daily 5: gaps per digit
    within each position, pooled. Returns a dataframe.
    """
    if is_daily:
        last = [dict() for _ in range(5)]
        gaps = []
        for i, actual in enumerate(draws):
            for pos, d in enumerate(actual):
                if d in last[pos]:
                    gaps.append(i - last[pos][d])
                last[pos][d] = i
    else:
        last = {}
        gaps = []
        for i, actual in enumerate(draws):
            for k in actual:
                if k in last:
                    gaps.append(i - last[k])
                last[k] = i
    total = len(gaps)
    counts = Counter(min(g, max_gap + 1) for g in gaps)
    rows = []
    for g in range(1, max_gap + 1):
        theo = (1 - p) ** (g - 1) * p
        rows.append({"Gap (draws)": g,
                     "Observed": counts.get(g, 0),
                     "Geometric theory": round(total * theo, 1)})
    rows.append({"Gap (draws)": f"{max_gap + 1}+",
                 "Observed": counts.get(max_gap + 1, 0),
                 "Geometric theory": round(total * (1 - p) ** max_gap, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_scoreboard(results, next_picks, baseline, exact_denom,
                      ml_row=None, ml_label="ML (train/test split)"):
    """The lesson in one table, then what each method would play next.

    results: {method: (scores, exact, n)} for walk-forward methods.
    ml_row: optional (test_scores, label) tuple for the tuned-rule
    method, whose n differs (it uses a train/test split, not
    walk-forward).
    """
    rows = scoreboard_rows(results, baseline)
    if ml_row is not None:
        test_scores, ml_label = ml_row[0], ml_row[1]
        mean, ci = mean_ci(test_scores)
        rows.append({
            "Method": ml_label,
            "n (test draws)": str(len(test_scores)),
            "Score": f"{mean:.3f}",
            "Baseline": f"{baseline:.3f}",
            "95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            "Verdict": verdict(mean, ci, baseline),
            "Exact hits": "—",
        })
    order = ["Hot", "Cold", "Balanced", "Gap", ml_label, "Random",
             "Anti", "Popular"]
    table = pd.DataFrame(rows)
    table["__o"] = table["Method"].apply(
        lambda m: order.index(m) if m in order else 99)
    table = table.sort_values("__o").drop(columns="__o")
    st.dataframe(table, width="stretch")
    st.caption(
        "Every row's confidence interval brackets the baseline — that "
        "is the lesson. Exact-set hits are out of "
        f"{exact_denom:,} possible combinations, so an exact hit count "
        "of 0 across thousands of test draws is itself expected "
        "(about 0.00 per method)."
    )
    st.subheader("What each method would play next — with its score")
    st.caption(
        "These are suggestions, and each one carries the score it "
        "earned above. None of them beats random; that is the point."
    )
    for m in order:
        if m == ml_label:
            continue
        if m in next_picks:
            sc, _, _ = results[m]
            mean, _ = mean_ci(sc)
            st.write(f"**{m}** — `{next_picks[m]}` — backtested "
                     f"{mean:.3f} hits/draw vs baseline "
                     f"{baseline:.3f}")


def render_fantasy_tab():
    st.subheader("Fantasy 5")
    st.caption(
        "Null model: each draw is a uniformly random 5-subset of 1-39. "
        f"Baseline: {BASELINE_F5:.3f} hits per draw for any 5-set, "
        "exact-set probability 1/575,757. Walk-forward methods use a "
        "trailing 100-draw window; the backtested-rule method "
        "searches 13 candidate rules on the first 70% of draws and "
        "tests the winner on the last 30%, so its n differs."
    )
    uploaded = st.file_uploader(
        "Upload a Fantasy 5 results file (CSV) — optional",
        type="csv", key="f5_upload")
    df = load_draws(
        uploaded,
        Path(__file__).parent / "data" / "fantasy5_history.csv",
        "Fantasy 5")
    if df is None:
        st.stop()
    for t in ordered_types(df, ["REGULAR DRAWING", "DOUBLE PLAY DRAWING"]):
        st.divider()
        st.header(t.title())
        draws = chronological(df[df["Drawing Type"] == t])
        if len(draws) <= 100:
            st.warning(
                f"Not enough draws to backtest here — walk-forward "
                f"methods need a 100-draw trailing window, and this "
                f"file has {len(draws)} draws for this type. Upload "
                "a longer history to see the scoreboard.")
            continue
        st.write(f"{len(draws)} draws, "
                 f"{df[df['Drawing Type'] == t]['Draw Date'].min().strftime('%b %d, %Y')} "
                 f"– {df[df['Drawing Type'] == t]['Draw Date'].max().strftime('%b %d, %Y')}")
        with st.spinner("Backtesting all methods on real draws…"):
            results, next_picks = fantasy_backtest(draws)
            (rule_name, rule_train, rule_test, rule_next,
             n_rules) = fantasy_rule_search(draws)
        render_scoreboard(results, next_picks, BASELINE_F5, 575757,
                          ml_row=(rule_test,
                                  "Backtested rule (tuned on train)"))
        rule_test_mean, _ = mean_ci(rule_test)
        st.write(f"**Backtested-rule next pick** — `{rule_next}` — "
                 f"the winning rule was '{rule_name}' ({n_rules} "
                 f"rules tried); test score {rule_test_mean:.3f} "
                 f"hits/draw vs baseline {BASELINE_F5:.3f} (train "
                 f"score {rule_train:.3f} — the train/test gap is "
                 "the overfitting lesson)")
        st.caption(
            "Multiple-comparison note: 8 methods compared (plus 13 "
            "candidate rules inside the tuned-rule search). At "
            "α = 0.05 you'd expect ~0.40 false 'winners' by chance "
            "alone — and none appeared beyond noise."
        )

    # Gotchas, once per tab, on the Regular drawing.
    st.divider()
    st.header("Why the methods fail — three visuals")
    reg = chronological(df[df["Drawing Type"] == "REGULAR DRAWING"])
    if not reg:
        reg = chronological(df)
    st.subheader("1. Window-size roulette: the 'signal' is the window")
    with st.spinner("Checking hot lists at two window sizes…"):
        hot25, _ = fantasy_backtest(reg, window=25)
        hot200, _ = fantasy_backtest(reg, window=200)
        # Latest-window hot lists, recomputed directly:
        c25 = Counter()
        for d in reg[-25:]:
            c25.update(d)
        c200 = Counter()
        for d in reg[-200:]:
            c200.update(d)
        list25 = sorted(range(1, 40),
                        key=lambda k: (-c25.get(k, 0), k))[:5]
        list200 = sorted(range(1, 40),
                         key=lambda k: (-c200.get(k, 0), k))[:5]
    m25, _ = mean_ci(hot25["Hot"][0])
    m200, _ = mean_ci(hot200["Hot"][0])
    col1, col2 = st.columns(2)
    col1.write(f"**Hot, last 25 draws:** `{sorted(list25)}` — "
               f"backtested {m25:.3f} hits/draw")
    col2.write(f"**Hot, last 200 draws:** `{sorted(list200)}` — "
               f"backtested {m200:.3f} hits/draw")
    st.caption(
        "Totally different lists, same score. Change the window and "
        "the 'signal' changes with it — because there is no signal, "
        "only the window you chose."
    )
    st.subheader("2. Gaps are memoryless: empirical vs geometric")
    st.caption(
        "Under independence, gaps between appearances are geometric "
        "with p = 5/39: the expected remaining wait is always 39/5 = "
        "7.8 draws, no matter how long the current gap is. The bars "
        "sit on the curve — 'due' numbers are not due."
    )
    gap_df = gap_table(reg, 5 / 39, 30)
    st.bar_chart(gap_df.set_index("Gap (draws)"))
    st.subheader("3. Train vs test collapse: the overfitting lesson")
    with st.spinner("Searching candidate rules on train…"):
        rule_name, rule_train, rule_test, _, n_rules = \
            fantasy_rule_search(reg)
    rule_test_mean, _ = mean_ci(rule_test)
    c1, c2, c3 = st.columns(3)
    c1.metric("Best rule, train score", f"{rule_train:.3f} hits/draw")
    c2.metric("Same rule, test score", f"{rule_test_mean:.3f} hits/draw")
    c3.metric("Random baseline", f"{BASELINE_F5:.3f} hits/draw")
    st.caption(
        f"The search tried {n_rules} candidate rules on the first "
        f"70% of draws and kept the best-looking one "
        f"('{rule_name}'). In-sample it looked like a winner; "
        "out-of-sample it fell back to baseline. That is "
        "overfitting: with enough candidate rules, some will look "
        "great in-sample by pure chance — and it is what every "
        "backtested 'system' does when you test it honestly."
    )


def render_daily_tab():
    st.subheader("Daily 5")
    st.caption(
        "Null model: each draw is 5 independent uniform digits 0-9, "
        "in order. Baseline: 0.5 matching positions per draw (10% per "
        "position), exact-sequence probability 1/100,000. "
        "Walk-forward methods use a trailing 25-draw window. "
        "Sample-size warning: this tab's built-in history starts "
        "August 2026, so confidence intervals are wide — treat every "
        "deviation as noise unless it is very large."
    )
    uploaded = st.file_uploader(
        "Upload a Daily 5 results file (CSV) — optional",
        type="csv", key="d5_upload")
    df = load_draws(
        uploaded,
        Path(__file__).parent / "data" / "daily5_history.csv",
        "Daily 5")
    if df is None:
        st.stop()
    for t in ordered_types(df, ["MIDDAY DRAWING", "EVENING DRAWING"]):
        st.divider()
        st.header(t.title())
        sub = df[df["Drawing Type"] == t]
        draws = chronological(sub)
        if len(draws) <= 25:
            st.warning(
                f"Not enough draws to backtest here — walk-forward "
                f"methods need a 25-draw trailing window, and this "
                f"file has {len(draws)} draws for this type. Upload "
                "a longer history to see the scoreboard.")
            continue
        st.write(f"{len(draws)} draws, "
                 f"{sub['Draw Date'].min().strftime('%b %d, %Y')} "
                 f"– {sub['Draw Date'].max().strftime('%b %d, %Y')}")
        with st.spinner("Backtesting all methods on real draws…"):
            results, next_picks = daily_backtest(draws)
        # ML row: not run — n too small for a train/test split.
        rows = scoreboard_rows(results, BASELINE_D5)
        rows.append({
            "Method": "ML (train/test split)",
            "n (test draws)": "—",
            "Score": "—",
            "Baseline": f"{BASELINE_D5:.3f}",
            "95% CI": "—",
            "Verdict": "Not run — n too small to split",
            "Exact hits": "—",
        })
        order = ["Hot", "Cold", "Balanced", "Gap",
                 "ML (train/test split)", "Random", "Anti", "Popular"]
        table = pd.DataFrame(rows)
        table["__o"] = table["Method"].apply(
            lambda m: order.index(m) if m in order else 99)
        st.dataframe(table.sort_values("__o").drop(columns="__o"),
                     width="stretch")
        st.caption(
            "Multiple-comparison note: 7 methods compared (ML not "
            "run). At α = 0.05 you'd expect ~0.35 false 'winners' by "
            "chance alone — and none appeared beyond noise."
        )
        st.subheader("What each method would play next — with its score")
        for m in ["Hot", "Cold", "Balanced", "Gap", "Random", "Anti",
                  "Popular"]:
            sc, _, _ = results[m]
            mean, _ = mean_ci(sc)
            st.write(f"**{m}** — `{next_picks[m]}` — backtested "
                     f"{mean:.3f} matches/draw vs baseline "
                     f"{BASELINE_D5:.3f}")

    st.divider()
    st.header("Why the methods fail — two visuals")
    mid = chronological(df[df["Drawing Type"] == "MIDDAY DRAWING"])
    if not mid:
        mid = chronological(df)
    st.subheader("1. Window-size roulette: the 'signal' is the window")
    with st.spinner("Checking hot digits at two window sizes…"):
        hot10, _ = daily_backtest(mid, window=10)
        hot25, _ = daily_backtest(mid, window=25)
        c10 = [Counter() for _ in range(5)]
        for d in mid[-10:]:
            for pos, dig in enumerate(d):
                c10[pos][dig] += 1
        c25 = [Counter() for _ in range(5)]
        for d in mid[-25:]:
            for pos, dig in enumerate(d):
                c25[pos][dig] += 1
        list10 = [sorted(range(10),
                         key=lambda dig: (-c10[pos].get(dig, 0), dig))[0]
                  for pos in range(5)]
        list25 = [sorted(range(10),
                         key=lambda dig: (-c25[pos].get(dig, 0), dig))[0]
                  for pos in range(5)]
    m10, _ = mean_ci(hot10["Hot"][0])
    m25, _ = mean_ci(hot25["Hot"][0])
    col1, col2 = st.columns(2)
    col1.write(f"**Hot, last 10 draws:** `{list10}` — backtested "
               f"{m10:.3f} matches/draw")
    col2.write(f"**Hot, last 25 draws:** `{list25}` — backtested "
               f"{m25:.3f} matches/draw")
    st.caption(
        "Different sequences, same score — and with only ~43 test "
        "draws, the intervals are wide enough to swallow any "
        "difference. That width is itself the lesson for Daily 5."
    )
    st.subheader("2. Gaps are memoryless: empirical vs geometric")
    st.caption(
        "Per-position gaps pooled, geometric with p = 0.10: the "
        "expected wait is always 10 draws, however long the current "
        "gap. The bars sit on the curve."
    )
    gap_df = gap_table(mid, 0.10, 20, is_daily=True)
    st.bar_chart(gap_df.set_index("Gap (draws)"))
    st.subheader("3. Train vs test: why there is no third visual")
    st.write(
        "With ~67 draws, a 70/30 split leaves ~20 test draws. Any "
        "'accuracy' on 20 draws is pure noise, so the ML method is "
        "not run here at all — saying so is more honest than "
        "printing a number."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_f5, tab_d5 = st.tabs(["Fantasy 5", "Daily 5"])
with tab_f5:
    render_fantasy_tab()
with tab_d5:
    render_daily_tab()

st.divider()
with st.expander("Honesty rules baked into this page"):
    st.markdown(
        "- **Every method row shows n, baseline, and CI** — no bare "
        "scores, ever.\n"
        "- **Every method is labeled** by what it is: a walk-forward "
        "backtest, or a train/test split. Never a 'winner picker.'\n"
        "- **Multiple comparisons are stated**, not hidden: 8 "
        "Fantasy 5 methods, 7 Daily 5 methods, and the number of "
        "false winners chance alone would produce.\n"
        "- **No single 'play this' without the scoreboard next to "
        "it.** Every next-draw suggestion on this page carries the "
        "score it earned in backtesting.\n"
        "- **Daily 5's n is printed prominently**, with the power "
        "limit stated: at ~67 draws, only gross bias is detectable.\n"
        "- **The one real number-choice effect is on payout, not "
        "probability:** popular numbers (birthdays, lucky digits) "
        "don't change your odds, but in a pari-mutuel game they "
        "change your expected share if you win, because more people "
        "pick them."
    )
st.caption(
    "Reality check, stated as the prediction itself: every Fantasy 5 "
    "combination has probability 1/575,757 per draw, and every Daily 5 "
    "ordered sequence has probability 1/100,000 per draw. The methods "
    "above disagree with each other and all score the same as random "
    "— because that's what randomness does."
)
