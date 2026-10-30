# Lottery Pattern Explorer — rebuilt 2026-10-30 after an external math
# critique. The old version ranked "most likely / least likely" picks;
# that framing implied prediction, and the critique showed it was
# descriptive statistics dressed as signal. This version does what the
# math actually supports: observed-vs-expected tables, exact
# combinatorial nulls, and goodness-of-fit fairness checks — with n and
# the expected value printed next to every count. It predicts nothing.
# Two games, two tabs, every stream kept separate (Fantasy 5 Regular /
# Double Play, Daily 5 Midday / Evening). Nothing is ever pooled.

import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lottery Pattern Explorer", layout="wide")
st.title("Lottery Pattern Explorer")
st.caption(
    "What past drawings actually looked like, compared against what fair, "
    "random drawings would look like. Every count below sits next to its "
    "expected value under the uniform null — and the honest prediction "
    "for the next draw is always the uniform one. Nothing here can tell "
    "you what to play."
)

# ---------------------------------------------------------------------------
# Chi-square p-values, pure Python (Numerical Recipes gammq: series for
# x < a+1, continued fraction otherwise). No scipy dependency, so the
# deployed requirements (streamlit, pandas) stay unchanged.
# ---------------------------------------------------------------------------

def _gser(a, x):
    gln = math.lgamma(a)
    ap, total, delta = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a, x):
    gln = math.lgamma(a)
    tiny = 1e-300
    b, c, d = x + 1 - a, 1 / tiny, 1 / (x + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_pvalue(chi2, df):
    """Survival function P(X >= chi2) for chi-square with df degrees."""
    if chi2 <= 0:
        return 1.0
    a, x = df / 2, chi2 / 2
    if x < a + 1:
        return 1 - _gser(a, x)
    return _gcf(a, x)


def chi2_test(observed, expected):
    """Return (chi2, df, p) for observed counts vs expected counts."""
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return chi2, df, chi2_pvalue(chi2, df)


def fairness_note(p):
    if p >= 0.05:
        return ("Consistent with fair, random draws. That is not proof of "
                "fairness — it means this test found nothing unusual.")
    return ("Unusual under the fair-draws null (p < 0.05). That is not "
            "proof of bias and not a betting edge: with many tests run, "
            "an occasional low p-value is expected by chance alone.")


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_draws(uploaded, builtin_path, game_name):
    """Return a parsed dataframe, or None after showing the upload help."""
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


def limit_input(key):
    return st.number_input(
        "How many draws back should we look? (per drawing type, most "
        "recent first — 0 = all)",
        min_value=0, step=50, value=0, key=key,
    )


# ---------------------------------------------------------------------------
# Fantasy 5 — 5 distinct numbers from 1-39, uniform 5-subset null
# ---------------------------------------------------------------------------

F5_N = 39
F5_K = 5
F5_COMBOS = math.comb(F5_N, F5_K)  # 575,757


def render_fantasy_section(draw_type, sub, limit):
    st.header(draw_type.title())
    draws = sub.sort_values("Draw Date", ascending=False)
    if limit:
        draws = draws.head(int(limit))
    n = len(draws)
    c1, c2, c3 = st.columns(3)
    c1.metric("Draws analyzed", n)
    c2.metric("Earliest", draws["Draw Date"].min().strftime("%b %d, %Y"))
    c3.metric("Latest", draws["Draw Date"].max().strftime("%b %d, %Y"))

    p = F5_K / F5_N
    expected_each = n * p
    sd_each = math.sqrt(n * p * (1 - p))
    freq = Counter()
    for nums in draws["Numbers"]:
        freq.update(nums)
    table = pd.DataFrame({
        "Number": list(range(1, F5_N + 1)),
        "Observed": [freq.get(k, 0) for k in range(1, F5_N + 1)],
        "Expected": [round(expected_each, 1)] * F5_N,
    })
    table["Difference"] = table["Observed"] - table["Expected"]
    table["z"] = ((table["Observed"] - expected_each) / sd_each).round(2)

    st.subheader("Observed vs expected — every number")
    st.caption(
        f"Each number is expected about {expected_each:.1f} times in "
        f"{n} draws (marginal SD about {sd_each:.1f}). A |z| near 2 is "
        "unremarkable: across 39 numbers, chance alone puts a couple "
        "of them that far out. This table describes the past; it does "
        "not rank what comes next."
    )
    st.dataframe(table, width="stretch")
    st.bar_chart(table.set_index("Number")["Observed"])

    chi2, df, pval = chi2_test(
        list(table["Observed"]), [expected_each] * F5_N)
    st.subheader("Fairness check — chi-square vs uniform")
    st.write(
        f"Chi-square = {chi2:.1f} on {df} degrees of freedom, "
        f"p = {pval:.3f}. {fairness_note(pval)} "
        f"With {n} draws this test can only catch gross bias; subtle "
        "bias would be invisible."
    )

    sums = [sum(nums) for nums in draws["Numbers"]]
    emp_mean = sum(sums) / n
    emp_sd = math.sqrt(sum((s - emp_mean) ** 2 for s in sums) / n)
    # Theoretical sum of 5 draws without replacement from 1..39:
    # mean 100, SD 23.8 (population variance (39^2-1)/12, finite-pop
    # correction (39-5)/(39-1)).
    theo_sd = math.sqrt(5 * (F5_N ** 2 - 1) / 12 * (F5_N - F5_K) / (F5_N - 1))
    st.subheader("Sum of the five numbers — observed vs exact theory")
    st.write(
        f"Observed mean {emp_mean:.1f}, SD {emp_sd:.1f}. "
        f"Exact theory for a uniform draw: mean 100.0, SD {theo_sd:.1f}. "
        "If the observed mean sits far from 100, that is a fairness "
        "question, not a picking signal."
    )

    odd_counts = Counter(sum(1 for x in nums if x % 2 == 1)
                         for nums in draws["Numbers"])
    odd_rows = []
    for k in range(6):
        theo_p = (math.comb(20, k) * math.comb(19, F5_K - k)
                  / F5_COMBOS)
        odd_rows.append({
            "Odd numbers in the draw": k,
            "Observed draws": odd_counts.get(k, 0),
            "Theoretical probability": round(theo_p, 4),
            "Expected draws": round(n * theo_p, 1),
        })
    st.subheader("Odd/even split — observed vs exact theory")
    st.caption(
        "Theory is hypergeometric: 20 odd and 19 even numbers in 1-39. "
        "Again, a mismatch here would be a fairness question, not an edge."
    )
    st.dataframe(pd.DataFrame(odd_rows), width="stretch")

    st.write(
        f"One random set, for comparison — it has the same "
        f"1-in-{F5_COMBOS:,} odds as every other set: "
        f"{sorted(random.sample(range(1, F5_N + 1), F5_K))}"
    )


def render_fantasy_tab():
    st.subheader("Fantasy 5")
    st.caption(
        "Null model: each draw is a uniformly random 5-subset of 1-39. "
        "Every combination has probability 1/575,757; every number has "
        "probability 5/39 per draw. That is also the only honest "
        "prediction for the next draw."
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
    limit = limit_input("f5_limit")
    for t in ordered_types(df, ["REGULAR DRAWING", "DOUBLE PLAY DRAWING"]):
        st.divider()
        sub = df[df["Drawing Type"] == t]
        render_fantasy_section(t, sub, limit)


# ---------------------------------------------------------------------------
# Daily 5 — 5 ordered digits 0-9, repeats allowed, i.i.d. uniform null
# ---------------------------------------------------------------------------

# Exact pattern counts out of 10^5 ordered sequences, derived by counting
# (and summing to exactly 100,000).
def _d5_pattern_probs():
    f = math.factorial
    counts = {
        "All distinct": 10 * 9 * 8 * 7 * 6,
        "Exactly one pair": 10 * math.comb(5, 2) * 9 * 8 * 7,
        "Two pairs": math.comb(10, 2) * (f(5) // (f(2) * f(2))) * 8,
        "Exactly three of a kind": 10 * math.comb(5, 3) * 9 * 8,
        "Full house (3 + 2)": 10 * 9 * (f(5) // (f(3) * f(2))),
        "Four of a kind": 10 * 5 * 9,
        "Five of a kind": 10,
    }
    assert sum(counts.values()) == 100000
    return {k: v / 100000 for k, v in counts.items()}


D5_PATTERN_PROBS = _d5_pattern_probs()


def _d5_pattern(digits):
    counts = sorted(Counter(digits).values(), reverse=True)
    if counts == [1, 1, 1, 1, 1]:
        return "All distinct"
    if counts == [2, 1, 1, 1]:
        return "Exactly one pair"
    if counts == [2, 2, 1]:
        return "Two pairs"
    if counts == [3, 1, 1]:
        return "Exactly three of a kind"
    if counts == [3, 2]:
        return "Full house (3 + 2)"
    if counts == [4, 1]:
        return "Four of a kind"
    return "Five of a kind"


def render_daily_section(draw_type, sub, limit):
    st.header(draw_type.title())
    draws = sub.sort_values("Draw Date", ascending=False)
    if limit:
        draws = draws.head(int(limit))
    n = len(draws)
    c1, c2, c3 = st.columns(3)
    c1.metric("Draws analyzed", n)
    c2.metric("Earliest", draws["Draw Date"].min().strftime("%b %d, %Y"))
    c3.metric("Latest", draws["Draw Date"].max().strftime("%b %d, %Y"))

    st.subheader("Digit counts by position — observed vs expected")
    st.caption(
        f"Each cell is expected about {n / 10:.1f} times in {n} draws "
        "(each position is uniform over 10 digits). With only "
        f"{n} draws, a digit showing up a few extra times is noise: "
        "the per-position SD is about "
        f"{math.sqrt(n * 0.1 * 0.9):.1f}, so a count of "
        f"{n / 10 + 2 * math.sqrt(n * 0.1 * 0.9):.0f} instead of "
        f"{n / 10:.0f} is unremarkable — and across 50 cells you "
        "should expect several such cells by chance."
    )
    pos_rows = {}
    for pos in range(5):
        counts = Counter(d[pos] for d in draws["Numbers"])
        pos_rows[f"Position {pos + 1}"] = [counts.get(dig, 0)
                                           for dig in range(10)]
    st.dataframe(
        pd.DataFrame(pos_rows, index=[f"Digit {d}" for d in range(10)]).T,
        width="stretch")

    pooled = Counter()
    for d in draws["Numbers"]:
        pooled.update(d)
    chi2, df, pval = chi2_test(
        [pooled.get(dig, 0) for dig in range(10)], [n / 2] * 10)
    st.subheader("Fairness check — pooled chi-square vs uniform")
    st.write(
        f"Pooling all 5 positions gives {5 * n} digit observations, "
        f"expected {n / 2:.1f} per digit. Chi-square = {chi2:.1f} on "
        f"{df} degrees of freedom, p = {pval:.3f}. "
        f"{fairness_note(pval)} "
        f"Power warning: with {n} draws you can only detect a digit "
        "running at roughly twice its expected rate; anything subtler "
        "is invisible here."
    )

    pat_counts = Counter(_d5_pattern(d) for d in draws["Numbers"])
    pat_rows = [{
        "Pattern": pat,
        "Observed draws": pat_counts.get(pat, 0),
        "Theoretical probability": round(prob, 4),
        "Expected draws": round(n * prob, 1),
    } for pat, prob in D5_PATTERN_PROBS.items()]
    st.subheader("Repeat patterns — observed vs exact theory")
    st.caption(
        "Theory is exact counting over all 100,000 ordered sequences. "
        "This is the most informative Daily 5 table: it shows whether "
        "the observed mix of repeats looks like fair draws."
    )
    st.dataframe(pd.DataFrame(pat_rows), width="stretch")

    most_observed = []
    for pos in range(5):
        counts = Counter(d[pos] for d in draws["Numbers"])
        most_observed.append(min(counts, key=lambda dig: (-counts[dig], dig)))
    st.write(
        "Most observed digit per position (description only — each "
        f"position is 10% per digit on the next draw): "
        f"{', '.join(str(d) for d in most_observed)}"
    )


def render_daily_tab():
    st.subheader("Daily 5")
    st.caption(
        "Null model: each draw is 5 independent uniform digits 0-9, in "
        "order. Every ordered sequence has probability 1/100,000; every "
        "digit has probability 10% in every position. That is also the "
        "only honest prediction for the next draw. Note the sample "
        "size: this tab's built-in history starts August 2026, so "
        "these tables are thin — treat every deviation as noise "
        "unless it is very large."
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
    limit = limit_input("d5_limit")
    for t in ordered_types(df, ["MIDDAY DRAWING", "EVENING DRAWING"]):
        st.divider()
        sub = df[df["Drawing Type"] == t]
        render_daily_section(t, sub, limit)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_f5, tab_d5 = st.tabs(["Fantasy 5", "Daily 5"])
with tab_f5:
    render_fantasy_tab()
with tab_d5:
    render_daily_tab()

st.divider()
with st.expander("How to read this page — the honest version"):
    st.markdown(
        "- **Two null models, stated up front.** Fantasy 5: uniform "
        "5-subsets of 1-39. Daily 5: five independent uniform digits "
        "0-9. Every 'expected' number on this page comes from those "
        "nulls, not from a trend.\n"
        "- **Descriptive vs test.** Counts and observed-vs-expected "
        "tables describe the past. The chi-square lines are fairness "
        "tests: they ask whether the past looks like fair draws. "
        "Neither predicts the next draw.\n"
        "- **Every count carries its n and its expected value.** A "
        "count without its null expectation is misleading by "
        "omission, so none is shown that way.\n"
        "- **Multiple comparisons.** Ranking 39 numbers (or 50 "
        "digit-cells) guarantees some will look extreme by chance. "
        "A |z| near 2 is not a signal.\n"
        "- **No 'due', 'overdue', or 'trend' language.** A number "
        "that hasn't hit recently is not more likely to hit next. "
        "Draws are independent.\n"
        "- **Streams are never pooled.** Regular vs Double Play, "
        "Midday vs Evening, Fantasy 5 vs Daily 5 — every stat is "
        "computed on one stream only.\n"
        "- **Sample-size honesty.** Fantasy 5 (~2,120 draws per type) "
        "can catch only gross bias. Daily 5 (~67 draws per type) can "
        "catch almost nothing. Both say so next to their tests."
    )
st.caption(
    "Reality check, stated as the prediction itself: every Fantasy 5 "
    "combination has probability 1/575,757 per draw, and every Daily 5 "
    "ordered sequence has probability 1/100,000 per draw. If any part "
    "of this page ever seems to promise more than that, it is wrong."
)
