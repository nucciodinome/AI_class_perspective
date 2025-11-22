import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations
from collections import Counter

# ------------------------------------------------------
# SAFE NLTK DOWNLOAD
# ------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)

# ------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Ultra-Advanced Text Analysis Suite", layout="wide")
st.title("📊 Ultra-Advanced Text Analysis Suite")

# ------------------------------------------------------
# UPLOAD FILE
# ------------------------------------------------------
uploaded = st.file_uploader("Upload dataset (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded:

    # Load file safely
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # --------------------------------------------------
    # SIDEBAR SETTINGS
    # --------------------------------------------------
    st.sidebar.header("Settings")

    text_col = st.sidebar.selectbox("Select text column", df.columns)
    model_col = st.sidebar.selectbox("Select model column", df.columns)
    region_col = st.sidebar.selectbox("Select user region column", df.columns)

    # -----------------------------------------
    # STOPWORDS (SAFE, EXTENDED)
    # -----------------------------------------
    default_stop = {
        "a","about","above","across","after","again","against","all","almost","alone","along","already",
        "also","although","always","am","among","an","and","another","any","anybody","anyone","anything",
        "anyway","anywhere","are","around","as","at","back","be","became","because","become","becomes",
        "becoming","been","before","beforehand","behind","being","below","beside","besides","between",
        "beyond","both","but","by","can","cannot","could","did","do","does","doing","done","down","during",
        "each","either","else","elsewhere","enough","even","ever","every","everybody","everyone",
        "everything","everywhere","except","few","fifteen","fifty","first","five","for","former",
        "formerly","forty","four","from","further","had","has","have","having","he","her","here","hers",
        "herself","him","himself","his","how","however","i","if","in","indeed","instead","into","is","it",
        "its","itself","just","keep","keeps","kept","know","known","knows","last","least","less","let",
        "likely","long","made","make","makes","many","may","maybe","me","might","mine","more","most",
        "mostly","much","must","my","myself","neither","never","nevertheless","new","next","nine","no",
        "nobody","none","nor","not","nothing","now","nowhere","of","off","often","on","once","one","only",
        "onto","or","other","others","otherwise","our","ours","ourselves","out","over","own","part","per",
        "perhaps","put","rather","really","said","same","say","second","see","seem","seemed","seeming",
        "seems","several","she","should","since","six","so","some","somebody","someone","something",
        "sometimes","somewhere","still","such","taking","ten","than","that","the","their","theirs","them",
        "themselves","then","there","therefore","these","they","thing","things","third","this","those",
        "though","three","through","throughout","to","together","too","toward","try","trying","twenty",
        "two","under","until","up","upon","us","use","used","usually","very","via","was","we","well","were",
        "what","whatever","when","whenever","where","whether","which","while","who","whoever","whole",
        "whom","why","will","with","within","without","would","yes","yet","you","your","yours","yourself",
        "yourselves",

        # Conversational filler
        "uh","um","hmm","ok","okay","yeah","yep","right","well","basically","literally","actually",
        "kinda","sorta","maybe","guess","just","really","quite",

        # Academic filler
        "analysis","study","studies","research","paper","text","section","paragraph","author","authors"
    }

    sociological_stop = {"people","think","say","study","student","analysis","text","use","like"}

    custom_stop = st.sidebar.text_area("Add custom stopwords (comma-separated)")
    custom_stop = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    stopwords_final = sorted(default_stop.union(sociological_stop).union(custom_stop))

    # --------------------------------------------------
    # Topic modeling sliders
    # --------------------------------------------------
    n_topics = st.sidebar.slider("Number of topic clusters", 3, 20, 6)
    min_df = st.sidebar.slider("Min document frequency", 1, 10, 2)
    max_df = st.sidebar.slider("Max document frequency fraction", 0.1, 1.0, 0.9)
    top_n_words = st.sidebar.slider("Top terms per topic", 5, 30, 10)

    # --------------------------------------------------
    # Create tabs
    # --------------------------------------------------
    tabs = st.tabs([
        "1️⃣ Topic Modeling",
        "2️⃣ STM-style Word Differences",
        "3️⃣ Topic Distance Map",
        "4️⃣ Semantic Network",
        "5️⃣ Sentiment",
        "6️⃣ Wordclouds",
        "7️⃣ Region × Model Analysis"
    ])

    # --------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------
    df = df.dropna(subset=[text_col])
    docs = df[text_col].astype(str).tolist()

    # --------------------------------------------------
    # TF-IDF (UNIGRAMS ONLY)
    # --------------------------------------------------
    tfidf = TfidfVectorizer(
        stop_words=stopwords_final,
        max_features=6000,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1,1)     # <<< ONLY WORDS, NO BIGRAMS
    )
    X = tfidf.fit_transform(docs)
    feature_names = np.array(tfidf.get_feature_names_out())

    # --------------------------------------------------
    # NMF TOPIC MODEL
    # --------------------------------------------------
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    df["topic"] = W.argmax(axis=1)

    # helper: top words
    def top_words(topic_idx, n=top_n_words):
        idx = H[topic_idx].argsort()[-n:][::-1]
        return feature_names[idx], H[topic_idx][idx]

    # ------------------------------------------------------------
    # TAB 1 — TOPIC MODELING
    # ------------------------------------------------------------
    with tabs[0]:
        st.subheader("📌 Extracted Topics")

        topic_rows = []
        for t in range(n_topics):
            words, _ = top_words(t)
            topic_rows.append({"Topic": t, "Top Terms": ", ".join(words)})

        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)

        # ------------------- Topic distribution by region
        st.subheader("Topic Distribution by Region")
        fig = px.histogram(
            df, x="topic", color=region_col, barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------- Topic-term heatmap
        st.subheader("Topic–Term Heatmap")
        heat = np.vstack([top_words(t, top_n_words)[1] for t in range(n_topics)])
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=[f"Top {i+1}" for i in range(top_n_words)],
                y=[f"Topic {i}" for i in range(n_topics)],
                colorscale="Viridis"
            )
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Brighter yellow = stronger association.")

    # ------------------------------------------------------------
    # TAB 2 — STM-STYLE DIFFERENCE PLOTS
    # ------------------------------------------------------------
    with tabs[1]:
        st.header("STM-Style Difference-in-Word-Use Analysis")

        condition = st.selectbox("Compare by:", [model_col, region_col])

        groups = df.groupby(condition)
        if len(groups) != 2:
            st.warning("Need exactly 2 groups to compute difference plot.")
        else:
            g1, g2 = list(groups.groups.keys())
            texts1 = " ".join(df[df[condition] == g1][text_col].astype(str))
            texts2 = " ".join(df[df[condition] == g2][text_col].astype(str))

            words1 = Counter([w for w in texts1.lower().split() if w not in stopwords_final])
            words2 = Counter([w for w in texts2.lower().split() if w not in stopwords_final])

            vocab = list(set(words1.keys()).union(words2.keys()))
            diff = []
            for w in vocab:
                diff.append({
                    "word": w,
                    "diff": words1[w] - words2[w]
                })

            diff_df = pd.DataFrame(diff)
            diff_df["abs"] = diff_df["diff"].abs()
            diff_df = diff_df.sort_values("abs", ascending=False).head(40)

            fig = px.bar(
                diff_df, x="word", y="diff",
                color="diff", color_continuous_scale="RdBu",
                title=f"Word Usage Difference: {g1} vs {g2}"
            )
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 3 — TOPIC DISTANCE MAP
    # ------------------------------------------------------------
    with tabs[2]:
        st.header("Topic Distance Map (MDS)")
        dist = pairwise_distances(H)
        coords = MDS(n_components=2, random_state=42,
                     dissimilarity="precomputed").fit_transform(dist)
        fig = px.scatter(
            x=coords[:,0], y=coords[:,1],
            text=[f"T{i}" for i in range(n_topics)],
            color=list(range(n_topics)),
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 4 — SEMANTIC NETWORK
    # ------------------------------------------------------------
    with tabs[3]:
        st.header("Semantic Network (Improved)")

        G = nx.Graph()

        for t in range(n_topics):
            words, _ = top_words(t, 12)
            for w1, w2 in combinations(words, 2):
                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

        centrality = nx.betweenness_centrality(G)
        size = [8 + 80 * centrality[n] for n in G.nodes()]

        pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y = [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)

        fig_net = go.Figure()

        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="lightgray")
        ))

        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=size, color=size, colorscale="Viridis")
        ))

        fig_net.update_layout(height=800,
                              margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_net, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 5 — SENTIMENT ANALYSIS
    # ------------------------------------------------------------
    with tabs[4]:
        st.header("Sentiment Analysis (VADER)")

        sia = SentimentIntensityAnalyzer()
        df["sentiment_score"] = df[text_col].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda s: "Positive" if s>0.05 else ("Negative" if s<-0.05 else "Neutral")
        )

        color_map = {
            "Positive": "#4DA6FF",   # soft blue
            "Negative": "#FF6666",   # soft red
            "Neutral":  "#BFBFBF"
        }

        st.subheader("Sentiment Distribution")
        fig = px.histogram(
            df, x="sentiment_label",
            color="sentiment_label",
            color_discrete_map=color_map,
            barnorm="percent"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment by Region")
        fig = px.histogram(
            df, x="sentiment_label", color=region_col,
            color_discrete_map=color_map, barnorm="percent"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment by Model")
        fig = px.histogram(
            df, x="sentiment_label", color=model_col,
            color_discrete_map=color_map, barnorm="percent"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 6 — WORDCLOUDS
    # ------------------------------------------------------------
    with tabs[5]:
        st.header("Wordclouds (Region × Model)")

        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.subheader(f"{reg} – {mod}")
            text = " ".join(subset[text_col].astype(str))
            if len(text) < 20:
                st.write("Not enough text.")
                continue
            wc = WordCloud(width=1000, height=500,
                           background_color="white").generate(text)
            st.image(wc.to_array(), use_container_width=True)

    # ------------------------------------------------------------
    # TAB 7 — REGION × MODEL INTERACTION
    # ------------------------------------------------------------
    with tabs[6]:
        st.header("Region × Model Interaction")

        fig = px.density_heatmap(
            df, x=region_col, y=model_col,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)


