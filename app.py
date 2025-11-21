import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations

# ---------------------------------------------
# SAFE NLTK DOWNLOAD
# ---------------------------------------------
import nltk
nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Advanced Text Analysis App", layout="wide")
st.title("Advanced Text Analysis App – Topics, Semantics, Sentiment, Wordclouds")

# ---------------------------------------------
# UPLOAD FILE
# ---------------------------------------------
uploaded = st.file_uploader("Upload dataset (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded:
    # Load file
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # -----------------------------------------
    # SIDEBAR INPUTS
    # -----------------------------------------
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

    # -----------------------------------------
    # TOPIC MODELING INPUTS
    # -----------------------------------------
    n_topics = st.sidebar.slider("Number of topics", 3, 20, 6)
    min_df = st.sidebar.slider("Min document frequency", 1, 10, 2)
    max_df = st.sidebar.slider("Max document frequency", 0.1, 1.0, 0.9)

    tabs = st.tabs([
        "Topic Modeling", "Topic Distance Map", "Semantic Network",
        "Sentiment", "Wordclouds", "Region × Model Analysis"
    ])

    # -----------------------------------------
    # CLEAN TEXT
    # -----------------------------------------
    df = df.dropna(subset=[text_col])
    docs = df[text_col].astype(str).tolist()

    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------
    tfidf = TfidfVectorizer(
        stop_words=stopwords_final,
        max_features=6000,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names_out()

    # -----------------------------------------
    # NMF TOPIC MODEL
    # -----------------------------------------
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    df["topic"] = W.argmax(axis=1)

    def label_topic(t):
        top_idx = H[t].argsort()[-6:][::-1]
        return ", ".join(feature_names[i] for i in top_idx)

    # -----------------------------------------
    # TAB 1 — TOPIC MODELING
    # -----------------------------------------
    with tabs[0]:
        st.header("Topic Modeling (NMF)")

        topic_labels = [label_topic(i) for i in range(n_topics)]
        st.dataframe(
            pd.DataFrame({"Topic": list(range(n_topics)), "Top Terms": topic_labels}),
            use_container_width=True
        )

        fig = px.histogram(df, x="topic", color=region_col, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        top_term_idx = np.argsort(H, axis=1)[:, -10:]
        matrix = np.array([[H[t][i] for i in top_term_idx[t]] for t in range(n_topics)])

        fig_hm = go.Figure(data=go.Heatmap(z=matrix, colorscale="Viridis"))
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("Brighter yellow = stronger association between topic and term.")

    # -----------------------------------------
    # TAB 2 — TOPIC DISTANCE MAP (MDS)
    # -----------------------------------------
    with tabs[1]:
        st.header("Topic Distance Map (MDS)")

        distances = pairwise_distances(H)
        coords = MDS(n_components=2, random_state=42,
                     dissimilarity="precomputed").fit_transform(distances)

        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            text=[f"T{i}" for i in range(n_topics)],
            color=list(range(n_topics)),
            color_continuous_scale="Viridis"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # TAB 3 — SEMANTIC NETWORK
    # -----------------------------------------
    with tabs[2]:
        st.header("Semantic Co-occurrence Network")

        G = nx.Graph()

        for t in range(n_topics):
            top_idx = H[t].argsort()[-7:]
            words = [feature_names[i] for i in top_idx]
            for w1, w2 in combinations(words, 2):
                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

        pos = nx.spring_layout(G, k=0.6, iterations=40)
        degrees = dict(G.degree())

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, size = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            size.append(10 + degrees[node] * 4)

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.7, color="lightgray")
        ))

        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=size, color=size, colorscale="Viridis")
        ))

        fig_net.update_layout(
            height=700,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )

        st.plotly_chart(fig_net, use_container_width=True)
        st.caption("Bigger nodes = more central concepts. Edges = co-occurrence in topic top terms.")

    # -----------------------------------------
    # TAB 4 — SENTIMENT
    # -----------------------------------------
    with tabs[3]:
        st.header("Sentiment Analysis (VADER)")

        sia = SentimentIntensityAnalyzer()
        df["sentiment_score"] = df[text_col].apply(lambda x: sia.polarity_scores(x)["compound"])
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda s: "Positive" if s > 0.05 else ("Negative" if s < -0.05 else "Neutral")
        )

        color_map = {"Positive": "blue", "Negative": "red", "Neutral": "gray"}

        fig = px.histogram(
            df, x="sentiment_label", color="sentiment_label",
            color_discrete_map=color_map, barnorm="percent"
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

    # -----------------------------------------
    # TAB 5 — WORDCLOUDS
    # -----------------------------------------
    with tabs[4]:
        st.header("Wordclouds (Region × Model)")

        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.subheader(f"{reg} – {mod}")

            text = " ".join(subset[text_col].astype(str).tolist())
            if len(text.strip()) < 5:
                st.write("Not enough text.")
                continue

            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), use_container_width=True)

    # -----------------------------------------
    # TAB 6 — REGION × MODEL ANALYSIS
    # -----------------------------------------
    with tabs[5]:
        st.header("Region × Model Interaction Analysis")

        fig = px.density_heatmap(
            df, x=region_col, y=model_col,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Shows concentration of responses for each Region × Model combination.")


