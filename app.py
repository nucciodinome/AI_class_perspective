import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from pyvis.network import Network
import tempfile
import os

nltk.download('vader_lexicon')

st.set_page_config(page_title="Advanced Text Analysis App", layout="wide")
st.title("Advanced Text Analysis App – NMF, Semantic Network, Sentiment, Wordclouds")

uploaded = st.file_uploader("Upload Excel dataset", type=["xlsx", "xls", "csv"])
if uploaded:
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    text_col = st.selectbox("Select text column", df.columns)
    model_col = st.selectbox("Select AI model column", df.columns)
    region_col = st.selectbox("Select user region column", df.columns)

    tabs = st.tabs(["Topic Modeling", "Semantic Network", "Sentiment Analysis", "Wordclouds"])

    df = df.dropna(subset=[text_col])
    docs = df[text_col].astype(str).tolist()

    ##########################
    # TOPIC MODELING TAB
    ##########################
    with tabs[0]:
        st.header("Topic Modeling with NMF")
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(docs)

        nmf = NMF(n_components=5, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_
        df['topic'] = W.argmax(axis=1)

        feature_names = tfidf.get_feature_names_out()
        topic_terms = []
        for t in range(5):
            top_idx = H[t].argsort()[-10:][::-1]
            terms = [feature_names[i] for i in top_idx]
            topic_terms.append({"topic": t, "terms": ", ".join(terms)})

        st.subheader("Top Terms per Topic")
        st.table(pd.DataFrame(topic_terms))

        st.subheader("Topic Distribution by User Region")
        fig = px.histogram(df, x='topic', color=region_col, barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    ##########################
    # SEMANTIC NETWORK TAB
    ##########################
    with tabs[1]:
        st.header("Semantic Network – Interactive")

        G = nx.Graph()
        for t in range(5):
            top_idx = H[t].argsort()[-8:]
            words = [feature_names[i] for i in top_idx]
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    weight = float(H[t][top_idx[i]] + H[t][top_idx[i+1]]) / 2
                    G.add_edge(w1, w2, weight=round(weight, 3))

        nt = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')
        nt.force_atlas_2based(gravity=-50)

        for node in G.nodes():
            degree = G.degree(node)
            nt.add_node(node, size=10 + degree * 3)

        for edge in G.edges(data=True):
            nt.add_edge(edge[0], edge[1], title=str(edge[2]['weight']))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        nt.save_graph(tmp.name)

        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)

        os.remove(tmp.name)

    ##########################
    # SENTIMENT TAB
    ##########################
    with tabs[2]:
        st.header("Sentiment Analysis (VADER)")
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df[text_col].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

        fig = px.histogram(df, x='sentiment_label', color=model_col, barnorm='percent')
        st.plotly_chart(fig, use_container_width=True)

    ##########################
    # WORDCLOUD TAB
    ##########################
    with tabs[3]:
        st.header("Wordclouds (User Region × AI Model)")
        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.write(f"### {reg} – {mod}")
            text = " ".join(subset[text_col].astype(str).tolist())
            wc = WordCloud(width=600, height=400, colormap='viridis').generate(text)
            fig_wc = plt.figure(figsize=(6, 4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig_wc)

