import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
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

col1, col2, col3 = st.columns(3)
col1.metric("Draws", len(df))
col2.metric("Earliest", df["Draw Date"].min().strftime("%b %d, %Y"))
col3.metric("Latest", df["Draw Date"].max().strftime("%b %d, %Y"))

draw_type = st.selectbox(
    "Drawing type",
    ["All", "REGULAR DRAWING", "DOUBLE PLAY DRAWING"]
)
subset = df if draw_type == "All" else df[df["Drawing Type"] == draw_type]

draws = subset.sort_values("Draw Date", ascending=False).to_dict("records")

def top5(scores, label):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return sorted(n for n, _ in ranked), ranked

# Method 1
freq = Counter()
for d in draws:
    freq.update(d["Numbers"])
m1, m1_scores = top5({n: freq.get(n, 0) for n in range(1, 40)}, "freq")

# Method 2
decay = 0.999
rw = defaultdict(float)
for i, d in enumerate(draws):
    w = decay ** i
    for n in d["Numbers"]:
        rw[n] += w
m2, m2_scores = top5({n: rw.get(n, 0) for n in range(1, 40)}, "rw")

# Method 3
pair_counts = Counter()
for d in draws:
    for p in combinations(sorted(d["Numbers"]), 2):
        pair_counts[p] += 1
cooc = {n: sum(c for (a, b), c in pair_counts.items() if a == n or b == n)
        for n in range(1, 40)}
m3, m3_scores = top5(cooc, "cooc")

# Method 4
def greedy_set(seed):
    chosen = list(seed)
    while len(chosen) < 5:
        best_n, best_gain = None, -1
        for cand in range(1, 40):
            if cand in chosen:
                continue
            gain = sum(pair_counts.get(tuple(sorted((cand, c))), 0) for c in chosen)
            if gain > best_gain:
                best_n, best_gain = cand, gain
        chosen.append(best_n)
    return sorted(chosen)

if pair_counts:
    seed = max(pair_counts, key=pair_counts.get)
    m4 = greedy_set(seed)
else:
    m4, seed = [1, 2, 3, 4, 5], (0, 0)

# Method 5
m5 = sorted(random.sample(range(1, 40), 5))

st.subheader("Picks by each method")
rows = [
    ("1. Raw frequency", m1, "All-time hottest numbers"),
    ("2. Recency-weighted", m2, "Recent draws count more"),
    ("3. Co-occurrence centrality", m3, "Frequency × 4 — not independent"),
    ("4. Greedy pair set", m4, f"Seeded from hottest pair {seed}"),
    ("5. Random baseline", m5, "Same real odds as the others"),
]
for label, nums, note in rows:
    st.markdown(f"**{label}** — `{nums}`  \n*{note}*")

st.subheader("Frequency chart")
freq_df = pd.DataFrame(
    {"Number": list(range(1, 40)),
     "Count": [freq.get(n, 0) for n in range(1, 40)]}
).set_index("Number")
st.bar_chart(freq_df)

with st.expander("Show all scores"):
    for label, nums, _ in rows[:3]:
        st.write(f"**{label}**")
        st.dataframe(pd.DataFrame(
            [{"Number": n, "Score": s} for n, s in
             (m1_scores if label.startswith("1") else
              m2_scores if label.startswith("2") else m3_scores)],
        ))

st.divider()
st.caption(
    "Every set has 1 in 575,757 odds. The methods disagree because they "
    "describe the past differently — none predicts the future."
)
