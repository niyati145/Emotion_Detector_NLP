"""
Emotion Detector: Streamlit UI for the "Emotions Dataset for NLP" notebook.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Put `train.txt` (format: text;emotion) next to this file, or upload it in the sidebar.
"""
import io
import string

import altair as alt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------------------------------------------------------------------
# Page setup and theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="wide")

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}
COLOR = {
    "joy": "#F5B700", "sadness": "#3D7EDB", "anger": "#E5484D",
    "fear": "#8E5BD9", "love": "#F06A9B", "surprise": "#1FB5A8",
}
EXAMPLES = [
    "i feel so happy and grateful for everything today",
    "i cannot believe how rude he was to me, i am furious",
    "i am terrified of what might happen tomorrow",
    "i miss her so much it hurts and i feel empty",
    "wow i did not expect that at all, what a shock",
    "i adore spending time with the people i care about",
]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 16px;
             background: linear-gradient(135deg, #6C5CE7 0%, #8E5BD9 50%, #F06A9B 100%);
             color: white; margin-bottom: 1.2rem;}
      .hero h1 {margin: 0; font-size: 2rem; color: white;}
      .hero p {margin: .3rem 0 0; opacity: .92;}
      .result-card {padding: 1.4rem; border-radius: 16px; text-align: center;
                    border: 2px solid var(--c); background: color-mix(in srgb, var(--c) 10%, transparent);}
      .result-card .emoji {font-size: 4rem; line-height: 1.1;}
      .result-card .label {font-size: 1.8rem; font-weight: 700; text-transform: capitalize; color: var(--c);}
      .result-card .conf {opacity: .75; font-size: 1rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
                                   padding: .8rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Preprocessing (same steps as the notebook)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:  # offline fallback
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(txt: str) -> str:
    stop_words = get_stopwords()
    txt = txt.lower()                                   # 1. lowercase
    txt = txt.translate(_PUNCT_TABLE)                   # 2. punctuation
    txt = "".join(c for c in txt if not c.isdigit())    # 3. numbers
    txt = "".join(c for c in txt if c.isascii())        # 6. emojis / non-ascii
    return " ".join(w for w in txt.split() if w not in stop_words)  # 7. stopwords


# ----------------------------------------------------------------------------
# Data + models
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), sep=";", header=None, names=["text", "emotion"]).dropna()
    df["clean"] = df["text"].apply(clean_text)
    return df


MODEL_NAMES = ["Logistic Regression (TF-IDF)", "Naive Bayes (TF-IDF)", "Naive Bayes (Bag of Words)"]


@st.cache_resource(show_spinner="Training models…")
def train_models(df: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["emotion"], test_size=0.2, random_state=42
    )
    bow = CountVectorizer().fit(X_tr)
    tfidf = TfidfVectorizer().fit(X_tr)

    models = {
        "Naive Bayes (Bag of Words)": (bow, MultinomialNB().fit(bow.transform(X_tr), y_tr)),
        "Naive Bayes (TF-IDF)": (tfidf, MultinomialNB().fit(tfidf.transform(X_tr), y_tr)),
        "Logistic Regression (TF-IDF)": (
            tfidf, LogisticRegression(max_iter=1000).fit(tfidf.transform(X_tr), y_tr)
        ),
    }
    metrics = {}
    for name, (vec, clf) in models.items():
        pred = clf.predict(vec.transform(X_te))
        metrics[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "cm": confusion_matrix(y_te, pred, labels=clf.classes_),
            "report": pd.DataFrame(
                classification_report(y_te, pred, output_dict=True, zero_division=0)
            ).T,
            "classes": list(clf.classes_),
        }
    return models, metrics


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Training data (train.txt)", type=["txt", "csv"],
                                help="Each line: text;emotion")
    model_name = st.selectbox("Model", MODEL_NAMES)
    st.divider()
    st.caption("Pipeline: lowercase → strip punctuation → strip digits → ASCII only → "
               "remove stopwords → vectorize → classify.")

raw = uploaded.getvalue() if uploaded else None
if raw is None:
    try:
        with open("train.txt", "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        pass

st.markdown(
    '<div class="hero"><h1>🎭 Emotion Detector</h1>'
    "<p>Type a sentence and see which emotion it expresses, how confident the model is, "
    "and which words drove the decision.</p></div>",
    unsafe_allow_html=True,
)

if raw is None:
    st.info("Upload **train.txt** in the sidebar (or place it next to `app.py`) to get started.")
    st.stop()

df = load_data(raw)
models, metrics = train_models(df)
vec, clf = models[model_name]
classes = list(clf.classes_)

tab_predict, tab_compare, tab_data = st.tabs(["🔮 Predict", "📊 Model comparison", "🗂️ Data explorer"])

# ----------------------------------------------------------------------------
# Tab 1: Predict
# ----------------------------------------------------------------------------
def set_example(example: str):
    st.session_state.text_input = example


with tab_predict:
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("Your text")
        st.text_area("Enter a sentence", key="text_input", height=140,
                     placeholder="e.g. i feel so happy and grateful today")
        st.caption("Try an example:")
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            cols[i % 3].button(ex[:28] + "…", key=f"ex{i}", width="stretch", help=ex,
                               on_click=set_example, args=(ex,))

    text = st.session_state.text_input.strip()

    with right:
        st.subheader("Result")
        if not text:
            st.info("Enter some text or pick an example to see a prediction.")
        else:
            cleaned = clean_text(text)
            if not cleaned:
                st.warning("After cleaning (stopwords, punctuation, etc.) nothing is left. "
                           "Try a longer sentence.")
            else:
                X = vec.transform([cleaned])
                proba = clf.predict_proba(X)[0]
                top = int(np.argmax(proba))
                label = classes[top]
                st.markdown(
                    f'<div class="result-card" style="--c:{COLOR.get(label, "#6C5CE7")}">'
                    f'<div class="emoji">{EMOJI.get(label, "🙂")}</div>'
                    f'<div class="label">{label}</div>'
                    f'<div class="conf">{proba[top]:.1%} confidence</div></div>',
                    unsafe_allow_html=True,
                )

    if text and clean_text(text):
        st.divider()
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**Probability by emotion**")
            pdf = pd.DataFrame({"emotion": classes, "probability": proba}).sort_values(
                "probability", ascending=False)
            chart = (
                alt.Chart(pdf).mark_bar(cornerRadiusEnd=6)
                .encode(
                    x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("emotion:N", sort="-x", title=None),
                    color=alt.Color("emotion:N", legend=None,
                                    scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values()))),
                    tooltip=["emotion", alt.Tooltip("probability:Q", format=".1%")],
                ).properties(height=240)
            )
            st.altair_chart(chart, width="stretch")

        with c2:
            st.markdown("**Words that influenced this prediction**")
            if isinstance(clf, LogisticRegression):
                contrib = X.toarray()[0] * clf.coef_[top]
            else:  # Naive Bayes: log-prob of word under predicted class vs. others
                lp = clf.feature_log_prob_
                contrib = X.toarray()[0] * (lp[top] - np.delete(lp, top, axis=0).mean(axis=0))
            names = np.array(vec.get_feature_names_out())
            idx = np.nonzero(X.toarray()[0])[0]
            if len(idx):
                wdf = pd.DataFrame({"word": names[idx], "impact": contrib[idx]})
                wdf = wdf.sort_values("impact", ascending=False).head(10)
                wchart = (
                    alt.Chart(wdf).mark_bar(cornerRadiusEnd=6, color=COLOR.get(label, "#6C5CE7"))
                    .encode(x=alt.X("impact:Q", title=f"push toward “{label}”"),
                            y=alt.Y("word:N", sort="-x", title=None),
                            tooltip=["word", alt.Tooltip("impact:Q", format=".3f")])
                    .properties(height=240)
                )
                st.altair_chart(wchart, width="stretch")
            else:
                st.caption("None of the words were seen during training.")

        with st.expander("See the cleaned text the model actually received"):
            st.code(cleaned)

# ----------------------------------------------------------------------------
# Tab 2: Model comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Accuracy on the 20% test split")
    cols = st.columns(len(MODEL_NAMES))
    best = max(metrics, key=lambda n: metrics[n]["accuracy"])
    for col, n in zip(cols, MODEL_NAMES):
        col.metric(n, f'{metrics[n]["accuracy"]:.2%}', "best" if n == best else None)

    acc_df = pd.DataFrame({"model": MODEL_NAMES,
                           "accuracy": [metrics[n]["accuracy"] for n in MODEL_NAMES]})
    st.altair_chart(
        alt.Chart(acc_df).mark_bar(cornerRadiusEnd=6, color="#6C5CE7").encode(
            x=alt.X("accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None),
            tooltip=["model", alt.Tooltip("accuracy:Q", format=".2%")],
        ).properties(height=150),
        width="stretch",
    )

    st.subheader(f"Details: {model_name}")
    m = metrics[model_name]
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown("**Confusion matrix**")
        cm = pd.DataFrame(m["cm"], index=m["classes"], columns=m["classes"])
        long = cm.reset_index().melt(id_vars="index", var_name="predicted", value_name="count")
        long = long.rename(columns={"index": "actual"})
        base = alt.Chart(long).encode(x=alt.X("predicted:N", title="Predicted"),
                                      y=alt.Y("actual:N", title="Actual"))
        heat = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="purples"),
                                                      legend=None))
        txt = base.mark_text(fontSize=13).encode(
            text="count:Q",
            color=alt.condition(alt.datum.count > long["count"].max() / 2,
                                alt.value("white"), alt.value("black")))
        st.altair_chart((heat + txt).properties(height=340), width="stretch")
    with d2:
        st.markdown("**Precision / recall / F1**")
        rep = m["report"].drop(index=["accuracy"], errors="ignore").round(3)
        st.dataframe(rep, width="stretch", height=340)

# ----------------------------------------------------------------------------
# Tab 3: Data explorer
# ----------------------------------------------------------------------------
with tab_data:
    k1, k2, k3 = st.columns(3)
    k1.metric("Samples", f"{len(df):,}")
    k2.metric("Emotions", df["emotion"].nunique())
    k3.metric("Avg. words / sample", f'{df["text"].str.split().str.len().mean():.1f}')

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Class distribution**")
        dist = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
        st.altair_chart(
            alt.Chart(dist).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("count:Q"), y=alt.Y("emotion:N", sort="-x", title=None),
                color=alt.Color("emotion:N", legend=None,
                                scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values()))),
                tooltip=["emotion", "count"]).properties(height=260),
            width="stretch")
    with g2:
        st.markdown("**Text length (words)**")
        lens = pd.DataFrame({"words": df["text"].str.split().str.len()})
        st.altair_chart(
            alt.Chart(lens).mark_bar(color="#8E5BD9").encode(
                x=alt.X("words:Q", bin=alt.Bin(maxbins=30), title="words"),
                y=alt.Y("count()", title="samples")).properties(height=260),
            width="stretch")

    st.markdown("**Most frequent words per emotion**")
    emo = st.selectbox("Emotion", sorted(df["emotion"].unique()),
                       format_func=lambda e: f"{EMOJI.get(e, '')} {e}")
    cv = CountVectorizer()
    counts = cv.fit_transform(df.loc[df["emotion"] == emo, "clean"]).sum(axis=0).A1
    wf = (pd.DataFrame({"word": cv.get_feature_names_out(), "count": counts})
          .sort_values("count", ascending=False).head(15))
    st.altair_chart(
        alt.Chart(wf).mark_bar(cornerRadiusEnd=6, color=COLOR.get(emo, "#6C5CE7")).encode(
            x="count:Q", y=alt.Y("word:N", sort="-x", title=None),
            tooltip=["word", "count"]).properties(height=380),
        width="stretch")

    with st.expander("Browse raw vs. cleaned samples"):
        st.dataframe(df.sample(min(200, len(df)), random_state=1)[["text", "clean", "emotion"]],
                     width="stretch", hide_index=True)