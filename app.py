import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import random

st.set_page_config(page_title="Lottery Pattern Explorer", layout="wide")
st.title("Lottery Pattern Explorer")
st.caption("A teaching tool: five ways to pick numbers, and why they all have the same odds.")

uploaded = st.file_uploader("Upload your CSV", type="csv")
if uploaded is None:
    st.info("Upload a CSV with columns: Draw Date, Winning Numbers, Drawing Type")
    st.stop()

df = pd.read_csv(uploaded, encoding="utf-8-sig")
df["Draw Date"] = pd.to_datetime(df["Draw Date"], format="%m/%d/%Y")
df["Numbers"] = df["Winning Numbers"].apply(
    lambda s: [int(x) for x in s.split(",")]
)

# Normalize the drawing-type labels so stray spaces or case differences in the
# CSV can never merge two different drawings into one pile.
if "Drawing Type" in df.columns:
    df["Drawing Type"] = df["Drawing Type"].astype(str).str.strip()
else:
    df["Drawing Type"] = "ALL DRAWINGS"

# Drawing types present in this file, regular drawing first, then double play,
# then anything else alphabetically.
PREFERRED_ORDER = ["REGULAR DRAWING", "DOUBLE PLAY DRAWING"]
present = list(df["Drawing Type"].unique())
ordered_types = [t for t in PREFERRED_ORDER if t in present]
ordered_types += sorted(t for t in present if t not in PREFERRED_ORDER)

st.subheader("What's in this file")
summary_cols = st.columns(len(ordered_types) + 1)
summary_cols[0].metric("Total draws", len(df))
for col, t in zip(summary_cols[1:], ordered_types):
    col.metric(t.title(), int((df["Drawing Type"] == t).sum()))

# How many draws each section analyzes: the most recent N of that drawing
# type, so Regular and Double Play are always compared on equal footing.
# Type any number — 0 means analyze all of them, and asking for more than
# a section has just uses everything it has.
limit = st.number_input(
    "Draws to analyze per drawing type (most recent first) — 0 = all",
    min_value=0,
    step=50,
    value=0,
)


def top5(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return sorted(n for n, _ in ranked), ranked


def bottom5(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1])[:5]
    return sorted(n for n, _ in ranked), ranked


BLUE = "#2563eb"
PINK = "#ec4899"


def pick_md(label, nums, note, color):
    """One pick line: only the numbers themselves are tinted."""
    return (f"**{label}** — <span style=\"color:{color}; "
            f"font-weight:700; font-family:monospace\">{nums}</span>"
            f"  \n*{note}*")


def analyze(draws):
    """Run all five methods on ONE drawing type's draws only."""
    # Method 1: raw frequency
    freq = Counter()
    for d in draws:
        freq.update(d["Numbers"])
    freq_scores = {n: freq.get(n, 0) for n in range(1, 40)}
    m1, m1_scores = top5(freq_scores)
    m1b, _ = bottom5(freq_scores)

    # Method 2: recency-weighted (draws are newest first)
    decay = 0.999
    rw = defaultdict(float)
    for i, d in enumerate(draws):
        w = decay ** i
        for n in d["Numbers"]:
            rw[n] += w
    rw_scores = {n: rw.get(n, 0) for n in range(1, 40)}
    m2, m2_scores = top5(rw_scores)
    m2b, _ = bottom5(rw_scores)

    # Method 3: overdue index — how long a number has gone without hitting
    # (its current drought) compared to its own usual gap between hits.
    # Score 2.0 means it has been missing twice as long as it usually goes.
    # (Pair counts are tallied here too because Method 4 builds on them.)
    pair_counts = Counter()
    for d in draws:
        for p in combinations(sorted(d["Numbers"]), 2):
            pair_counts[p] += 1
    total = len(draws)
    overdue = {}
    for n in range(1, 40):
        gap = 0
        for d in draws:  # draws are newest first
            if n in d["Numbers"]:
                break
            gap += 1
        hits = freq.get(n, 0)
        usual_gap = total / hits if hits else total
        overdue[n] = round(gap / usual_gap, 2)
    m3, m3_scores = top5(overdue)
    overdue_leader = max(overdue, key=overdue.get)
    m3b, _ = bottom5(overdue)
    freshest = min(overdue, key=overdue.get)

    # Method 4: greedy pair set
    def greedy_set(seed):
        chosen = list(seed)
        while len(chosen) < 5:
            best_n, best_gain = None, -1
            for cand in range(1, 40):
                if cand in chosen:
                    continue
                gain = sum(pair_counts.get(tuple(sorted((cand, c))), 0)
                           for c in chosen)
                if gain > best_gain:
                    best_n, best_gain = cand, gain
            chosen.append(best_n)
        return sorted(chosen)

    if pair_counts:
        seed = max(pair_counts, key=pair_counts.get)
        m4 = greedy_set(seed)
    else:
        m4, seed = [1, 2, 3, 4, 5], (0, 0)

    # Method 4, opposite: a greedy set built on the weakest pairs — start
    # from the pair that has hit together least (often never) and keep
    # adding the number that pairs weakest with those already chosen.
    def greedy_set_min(seed_pair):
        chosen = list(seed_pair)
        while len(chosen) < 5:
            best_n, best_gain = None, None
            for cand in range(1, 40):
                if cand in chosen:
                    continue
                gain = sum(pair_counts.get(tuple(sorted((cand, c))), 0)
                           for c in chosen)
                if best_gain is None or gain < best_gain:
                    best_n, best_gain = cand, gain
            chosen.append(best_n)
        return sorted(chosen)

    if pair_counts:
        seed_min = min(combinations(range(1, 40), 2),
                       key=lambda p: (pair_counts.get(p, 0), p))
        m4b = greedy_set_min(seed_min)
    else:
        m4b, seed_min = [1, 2, 3, 4, 5], (0, 0)

    # Method 5: random baseline
    m5 = sorted(random.sample(range(1, 40), 5))

    rows = [
        ("1. Raw frequency", m1, "All-time hottest numbers"),
        ("2. Recency-weighted", m2, "Recent draws count more"),
        ("3. Overdue index", m3,
         f"Most overdue: {overdue_leader} at {overdue[overdue_leader]}× its usual gap"),
        ("4. Greedy pair set", m4, f"Seeded from hottest pair {seed}"),
        ("5. Random baseline", m5, "Same real odds as the others"),
    ]
    # The opposites: same scores, read from the bottom instead of the top.
    # (Method 5 has no opposite — one random set is as likely as another.)
    opp_rows = [
        ("1. Raw frequency — coldest", m1b, "Least-hit numbers in the window"),
        ("2. Recency-weighted — coldest", m2b, "Lowest recency-weighted scores"),
        ("3. Overdue index — freshest", m3b,
         f"Freshest: {freshest} at {overdue[freshest]}× its usual gap"),
        ("4. Greedy pair set — weakest pairs", m4b,
         f"Seeded from weakest pair {seed_min}"),
    ]
    return rows, opp_rows, freq, (m1_scores, m2_scores, m3_scores)


def render_section(draw_type, sub):
    st.header(draw_type.title())
    draws = sub.sort_values("Draw Date", ascending=False).to_dict("records")

    c1, c2, c3 = st.columns(3)
    c1.metric("Draws analyzed", len(sub))
    c2.metric("Earliest", sub["Draw Date"].min().strftime("%b %d, %Y"))
    c3.metric("Latest", sub["Draw Date"].max().strftime("%b %d, %Y"))

    rows, opp_rows, freq, score_sets = analyze(draws)

    # Most likely on the left, least likely on the right, side by side.
    # Only the numbers are colored: blue on the left, pink on the right.
    left, right = st.columns(2)
    with left:
        st.subheader("Picks by each method")
        for label, nums, note in rows:
            st.markdown(pick_md(label, nums, note, BLUE),
                        unsafe_allow_html=True)
    with right:
        st.subheader("Opposite picks — least likely by the same math")
        for label, nums, note in opp_rows:
            st.markdown(pick_md(label, nums, note, PINK),
                        unsafe_allow_html=True)

    st.subheader("Frequency chart")
    freq_df = pd.DataFrame(
        {"Number": list(range(1, 40)),
         "Count": [freq.get(n, 0) for n in range(1, 40)]}
    ).set_index("Number")
    st.bar_chart(freq_df)

    with st.expander("Show all scores"):
        for (label, _, _), scores in zip(rows[:3], score_sets):
            st.write(f"**{label}**")
            st.dataframe(pd.DataFrame(
                [{"Number": n, "Score": s} for n, s in scores]))


# One separate section per drawing type — the regular drawing and the double
# play are never pooled together. Each section gets its own most-recent draws
# up to the limit picked above (all of them if the type has fewer).
for i, t in enumerate(ordered_types):
    if i > 0:
        st.divider()
    sub = df[df["Drawing Type"] == t].sort_values("Draw Date", ascending=False)
    if limit:
        sub = sub.head(int(limit))
    render_section(t, sub)

st.divider()
st.caption(
    "Every set has 1 in 575,757 odds. The methods disagree because they "
    "describe the past differently — none predicts the future."
)
