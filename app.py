import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx

st.title("Mini Text Analysis App")

uploaded = st.file_uploader("Upload Excel dataset", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded)
    text_col = st.selectbox("Select text column", df.columns)
    model_col = st.selectbox("Select model column", df.columns)
    region_col = st.selectbox("Select region column", df.columns)
    docs = df[text_col].astype(str).tolist()

    if st.button("Run Topic Model"):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(docs, show_progress_bar=True)
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings)
        df['topic'] = topics

        st.subheader("Estimate Effect by User Region")
        fig = px.histogram(df, x='topic', color=region_col, barmode='group')
        st.plotly_chart(fig)

        st.subheader("Semantic Network by User Region")
        G = nx.Graph()
        for t in df['topic'].unique():
            words = topic_model.get_topic(t)
            if words:
                for w, _ in words[:5]:
                    G.add_edge(f"topic_{t}", w)
        fig_net = plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500)
        st.pyplot(fig_net)

        st.subheader("Sentiment Bar Chart by AI Model")
        from transformers import pipeline
        sentiment = pipeline("sentiment-analysis")
        df['sentiment'] = df[text_col].apply(lambda x: sentiment(x)[0]['label'])
        fig2 = px.histogram(df, x='sentiment', color=model_col)
        st.plotly_chart(fig2)

        st.subheader("Word Clouds (4 combinations)")
        combos = df.groupby([region_col, model_col])
        for (reg, mod), subset in combos:
            wc = WordCloud(width=400, height=300).generate(" ".join(subset[text_col]))
            st.write(f"Region: {reg} | Model: {mod}")
            fig_wc = plt.figure()
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig_wc)
