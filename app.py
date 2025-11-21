import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations

nltk.download('vader_lexicon')

st.set_page_config(page_title="Advanced Text Analysis App", layout="wide")
st.title("Advanced Text Analysis App – Topics, Semantics, Sentiment, Wordclouds")

uploaded = st.file_uploader("Upload dataset (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded:
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.sidebar.header("Settings")
    text_col = st.sidebar.selectbox("Select text column", df.columns)
    model_col = st.sidebar.selectbox("Select model column", df.columns)
    region_col = st.sidebar.selectbox("Select user region column", df.columns)

    default_stop = set(nltk.corpus.stopwords.words('english')) if nltk else set()
    sociological_stop = {"people", "think", "say", "study", "student", "analysis", "text", "use", "like"}
    custom_stop = st.sidebar.text_area("Add custom stopwords (comma-separated)")
    custom_stop = set([w.strip().lower() for w in custom_stop.split(",") if w.strip()])
    stopwords_final = default_stop.union(sociological_stop).union(custom_stop)

    n_topics = st.sidebar.slider("Number of topics", 3, 20, 5)
    min_df = st.sidebar.slider("Min document frequency", 1, 10, 2)
    max_df = st.sidebar.slider("Max document frequency (fraction)", 0.1, 1.0, 0.9)

    tabs = st.tabs([
        "Topic Modeling", "Topic Distance Map", "Semantic Network", "Sentiment",
        "Wordclouds", "Region × Model Analysis"])

    df = df.dropna(subset=[text_col])
    docs = df[text_col].astype(str).tolist()

    tfidf = TfidfVectorizer(stop_words=stopwords_final, max_features=7000,
                            min_df=min_df, max_df=max_df, ngram_range=(1, 2))
    X = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    df['topic'] = W.argmax(axis=1)

    def label_topic(topic_idx):
        top_idx = H[topic_idx].argsort()[-5:][::-1]
        return ", ".join([feature_names[i] for i in top_idx])

    ###########################
    # TAB 1: TOPIC MODELING
    ###########################
    with tabs[0]:
        st.header("Topic Modeling (NMF)")
        st.write("This section shows the extracted topics and their most important terms. Higher weights indicate terms that define the topic.")

        topic_labels = [label_topic(i) for i in range(n_topics)]
        st.subheader("Top Terms per Topic")
        st.dataframe(pd.DataFrame({"Topic": range(n_topics), "Top Terms": topic_labels}), use_container_width=True)

        st.write("Interpretation: Each topic is a cluster of words frequently appearing together. Students should read the top terms to understand the theme.")

        st.subheader("Topic Distribution by User Region")
        fig = px.histogram(df, x='topic', color=region_col, barmode='group',
                           color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Interpretation: This chart shows which topics are more common in which user regions.")

        st.subheader("Topic-Term Heatmap")
        top_term_indices = np.argsort(H, axis=1)[:, -10:]
        heatmap_matrix = np.array([[H[t][i] for i in top_term_indices[t]] for t in range(n_topics)])
        fig_hm = go.Figure(data=go.Heatmap(z=heatmap_matrix, colorscale='Viridis'))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Interpretation: Darker cells indicate stronger relationships between terms and topics.")

    ###########################
    # TAB 2: TOPIC DISTANCE MAP
    ###########################
    with tabs[1]:
        st.header("Topic Distance Map (MDS)")
        st.write("This plot shows how similar or different the topics are. Closer points mean more similar topics.")

        distances = pairwise_distances(H)
        coords = MDS(n_components=2, random_state=42, dissimilarity='precomputed').fit_transform(distances)

        fig = px.scatter(x=coords[:, 0], y=coords[:, 1], text=[f"T{i}" for i in range(n_topics)],
                         color=list(range(n_topics)), color_continuous_scale='Viridis')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    ###########################
    # TAB 3: SEMANTIC NETWORK
    ###########################
    with tabs[2]:
        st.header("Semantic Co-occurrence Network")
        st.write("Nodes represent words. Links show how often they appear in the same topic.")

        G = nx.Graph()
        for t in range(n_topics):
            top_idx = H[t].argsort()[-8:]
            words = [feature_names[i] for i in top_idx]
            for w1, w2 in combinations(words, 2):
                weight = 1
                if G.has_edge(w1, w2):
                    G[w1][w2]['weight'] += weight
                else:
                    G.add_edge(w1, w2, weight=weight)

        degrees = dict(G.degree())
        pos = nx.spring_layout(G, k=0.4)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, sizes = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            sizes.append(10 + degrees[node] * 3)

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='lightgray')))
        fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                                     textposition='top center', marker=dict(size=sizes, color=sizes, colorscale='Viridis')))
        fig_net.update_layout(showlegend=False)
        st.plotly_chart(fig_net, use_container_width=True)

        st.caption("Interpretation: Bigger nodes are more central terms. Clusters reveal conceptual groupings.")

    ###########################
    # TAB 4: SENTIMENT ANALYSIS
    ###########################
    with tabs[3]:
        st.header("Sentiment Analysis (VADER)")
        st.write("This shows whether the text is expressed positively, negatively, or neutrally.")

        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df[text_col].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

        color_map = {"Positive": "blue", "Negative": "red", "Neutral": "gray"}
        fig = px.histogram(df, x='sentiment_label', color='sentiment_label',
                           color_discrete_map=color_map, barnorm='percent')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Interpretation: Compare emotional tone across models or regions.")

    ###########################
    # TAB 5: WORDCLOUDS
    ###########################
    with tabs[4]:
        st.header("Wordclouds (Region × Model)")
        st.write("These wordclouds highlight common terms within each Region–Model combination.")

        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.subheader(f"{reg} – {mod}")
            text = " ".join(subset[text_col].astype(str).tolist())

