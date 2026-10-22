import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import random

st.set_page_config(page_title="Lottery Pattern Explorer", layout="wide")
st.title("Lottery Pattern Explorer")
st.caption(
    "New here? This tool looks at past Michigan Fantasy 5 drawings and "
    "shows which numbers have been hot, cold, or overdue — six different "
    "ways, plus their opposites. It's pattern-spotting for fun: no list "
    "here can predict the next draw."
)

uploaded = st.file_uploader(
    "Upload a Fantasy 5 results file (CSV) — optional", type="csv")

# The app opens on its built-in results (every drawing back to January
# 2021, kept current by a nightly update). Uploading a file is optional
# and takes over whenever it's used.
BUILTIN_DATA = Path(__file__).parent / "data" / "fantasy5_history.csv"
if uploaded is not None:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")
elif BUILTIN_DATA.exists():
    df = pd.read_csv(BUILTIN_DATA, encoding="utf-8-sig")
    st.info(
        "You're looking at the built-in results — every Fantasy 5 "
        "drawing back to **January 2021**, updated nightly. Want even "
        "more history? Upload your own file above and it takes over."
    )
else:
    st.info(
        "**How it works:**\n\n"
        "1. Download a results file from michiganlottery.com → Winning "
        "Numbers → Historic Winning Numbers (pick Fantasy 5 and your dates).\n"
        "2. Upload it here.\n"
        "3. You get the Regular Drawing and the Double Play analyzed "
        "separately — hottest, coldest, and most overdue numbers for each.\n\n"
        "The file needs three columns: Draw Date, Winning Numbers, "
        "Drawing Type — exactly what the lottery site gives you."
    )
    st.stop()
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
st.caption(
    "Your file is split into the sections below — one per drawing type — "
    "and each is analyzed on its own. Nothing is mixed together."
)

# How many draws each section analyzes: the most recent N of that drawing
# type, so Regular and Double Play are always compared on equal footing.
# Type any number — 0 means analyze all of them, and asking for more than
# a section has just uses everything it has.
limit = st.number_input(
    "How many draws back should we look? (per drawing type, most recent "
    "first — 0 = all)",
    min_value=0,
    step=50,
    value=0,
)

with st.expander("New to this page? How to read it"):
    st.markdown(
        "- **Two drawings, two sections.** Fantasy 5 has a main drawing "
        "(Regular) and a bonus drawing (Double Play) every night. They "
        "have separate number histories, so they're never mixed.\n"
        "- **Left side (blue numbers) = most likely.** Five ways of "
        "picking the numbers that stand out most in past draws.\n"
        "- **Right side (pink numbers) = least likely.** The exact "
        "opposite of each method — the coldest, the freshest, the "
        "weakest pairs.\n"
        "- **The number box above** decides how far back to look, per "
        "drawing type. 0 looks at everything in your file.\n"
        "- **Reality check:** every 5-number combination has the same "
        "1 in 575,757 chance. These lists describe the past; none of "
        "them predicts the future."
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
    """Run all six methods on ONE drawing type's draws only."""
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

    # Method 6: hot + overdue blend — frequency and the overdue index,
    # each scaled 0-1, then added together. Rewards numbers that both
    # hit often AND are currently due.
    max_freq = max(freq_scores.values())
    max_overdue = max(overdue.values())
    blend = {
        n: freq_scores[n] / max_freq + overdue[n] / max_overdue
        for n in range(1, 40)
    }
    m6, _ = top5(blend)
    m6b, _ = bottom5(blend)

    rows = [
        ("1. Hottest numbers", m1,
         "Drawn most often in this window (raw frequency)"),
        ("2. Hot lately", m2,
         "Recent draws count more than old ones (recency-weighted)"),
        ("3. Most overdue", m3,
         f"Missing longest vs their usual rhythm (overdue index) — "
         f"{overdue_leader} is {overdue[overdue_leader]}× past due"),
        ("4. Best pair team", m4,
         f"Built from the pair that hits together most: {seed} "
         f"(greedy pair set)"),
        ("5. Pure random", m5,
         "A random set — same real odds as every other row"),
        ("6. Hot + overdue mix", m6,
         "Methods 1 + 3 blended — hotness and overdue, each scaled "
         "0–1 and added: proven hitters that are also due"),
    ]
    # The opposites: same scores, read from the bottom instead of the top.
    # (Method 5 has no opposite — one random set is as likely as another.)
    opp_rows = [
        ("1. Coldest numbers", m1b,
         "Drawn least often in this window (raw frequency, flipped)"),
        ("2. Coldest lately", m2b,
         "Lowest recent-draw scores (recency-weighted, flipped)"),
        ("3. Just hit", m3b,
         f"Freshest numbers (lowest overdue index) — {freshest} just "
         f"hit, at {overdue[freshest]}× its usual gap"),
        ("4. Weakest pair team", m4b,
         f"Built from the pair that hits together least: {seed_min} "
         f"(greedy pair set, flipped)"),
        ("5. Cold + fresh mix", m6b,
         "The blend flipped — coldest and freshest at once "
         "(frequency + overdue, reversed)"),
    ]
    return rows, opp_rows, freq, (m1_scores, m2_scores, m3_scores)


def render_section(draw_type, sub):
    st.header(draw_type.title())
    if draw_type == "REGULAR DRAWING":
        st.caption("The main nightly Fantasy 5 drawing.")
    elif draw_type == "DOUBLE PLAY DRAWING":
        st.caption("The bonus drawing held right after the main one — "
                   "its own separate number history.")
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
        st.subheader("Most likely — six ways to pick")
        for label, nums, note in rows:
            st.markdown(pick_md(label, nums, note, BLUE),
                        unsafe_allow_html=True)
    with right:
        st.subheader("Least likely — the same math, flipped")
        for label, nums, note in opp_rows:
            st.markdown(pick_md(label, nums, note, PINK),
                        unsafe_allow_html=True)

    st.subheader("How often each number hit")
    freq_df = pd.DataFrame(
        {"Number": list(range(1, 40)),
         "Count": [freq.get(n, 0) for n in range(1, 40)]}
    ).set_index("Number")
    st.bar_chart(freq_df)

    with st.expander("See the full scores behind methods 1–3"):
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
    "Reality check: every set of 5 numbers has the same 1 in 575,757 "
    "odds. The methods disagree because they describe the past "
    "differently — none of them predicts the future."
)
