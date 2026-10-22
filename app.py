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
limit = st.selectbox(
    "Draws to analyze per drawing type (most recent first)",
    ["All", 100, 250, 500, 750, 1000],
)


def top5(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return sorted(n for n, _ in ranked), ranked


def analyze(draws):
    """Run all five methods on ONE drawing type's draws only."""
    # Method 1: raw frequency
    freq = Counter()
    for d in draws:
        freq.update(d["Numbers"])
    m1, m1_scores = top5({n: freq.get(n, 0) for n in range(1, 40)})

    # Method 2: recency-weighted (draws are newest first)
    decay = 0.999
    rw = defaultdict(float)
    for i, d in enumerate(draws):
        w = decay ** i
        for n in d["Numbers"]:
            rw[n] += w
    m2, m2_scores = top5({n: rw.get(n, 0) for n in range(1, 40)})

    # Method 3: co-occurrence centrality
    pair_counts = Counter()
    for d in draws:
        for p in combinations(sorted(d["Numbers"]), 2):
            pair_counts[p] += 1
    cooc = {n: sum(c for (a, b), c in pair_counts.items() if a == n or b == n)
            for n in range(1, 40)}
    m3, m3_scores = top5(cooc)

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

    # Method 5: random baseline
    m5 = sorted(random.sample(range(1, 40), 5))

    rows = [
        ("1. Raw frequency", m1, "All-time hottest numbers"),
        ("2. Recency-weighted", m2, "Recent draws count more"),
        ("3. Co-occurrence centrality", m3, "Frequency × 4 — not independent"),
        ("4. Greedy pair set", m4, f"Seeded from hottest pair {seed}"),
        ("5. Random baseline", m5, "Same real odds as the others"),
    ]
    return rows, freq, (m1_scores, m2_scores, m3_scores)


def render_section(draw_type, sub):
    st.header(draw_type.title())
    draws = sub.sort_values("Draw Date", ascending=False).to_dict("records")

    c1, c2, c3 = st.columns(3)
    c1.metric("Draws analyzed", len(sub))
    c2.metric("Earliest", sub["Draw Date"].min().strftime("%b %d, %Y"))
    c3.metric("Latest", sub["Draw Date"].max().strftime("%b %d, %Y"))

    rows, freq, score_sets = analyze(draws)

    st.subheader("Picks by each method")
    for label, nums, note in rows:
        st.markdown(f"**{label}** — `{nums}`  \n*{note}*")

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
    if limit != "All":
        sub = sub.head(int(limit))
    render_section(t, sub)

st.divider()
st.caption(
    "Every set has 1 in 575,757 odds. The methods disagree because they "
    "describe the past differently — none predicts the future."
)
