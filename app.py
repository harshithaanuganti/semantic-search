#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Harshitha Anuganti
"""

import streamlit as st
import pickle as pkl
from sentence_transformers import SentenceTransformer, util
import regex as re
import torch
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud

st.title("Phuket Hotels' Review")
st.markdown("Harshitha Anuganti | MSBA 6490 | Assignment2 | v1.0")
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)

def load_data():
    with open("df.pkl", "rb") as file1:
        df = pkl.load(file1)
    return df

def load_model():
    with open("model.pkl", "rb") as file1:
        model = pkl.load(file1)
    return model

def load_corpus():
    with open("corpus_embeddings.pkl", "rb") as file1:
        corpus_embeddings = pkl.load(file1)
    with open("corpus.pkl", "rb") as file1:
        corpus = pkl.load(file1)
    return corpus, corpus_embeddings

def get_content(string):
    string_pattern = r"(?<=    )(.*)"
    # compile string pattern to re.Pattern object
    regex_pattern = re.compile(string_pattern)
    return str(regex_pattern.findall(string)[0])

def run():
    query = st.text_input("Search Here:", "Phuket")
    query_embeddings = model.encode(query, convert_to_tensor=True)
    top_k = min(5, len(corpus))

    cos_scores = util.pytorch_cos_sim(query_embeddings, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    st.write("\n\n======================\n\n")
    st.write("Query:", query)
    st.write("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        row_dict = df.loc[df['all_review']== corpus[idx]]
        st.write("Hotel:" , get_content(str(row_dict['hotelName'])))
        st.write(row_dict['sum_review'], "(Score: {:.4f})".format(score),"\n")

        wordcloud.generate(str(row_dict['all_review']))
        # create a figure
        plt.figure(figsize = (8, 8), facecolor = None)
        # add interpolation = bilinear to smooth things out
        plt.imshow(wordcloud)
        # and remove the axis
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        st.pyplot()


df = load_data()
corpus, corpus_embeddings= load_corpus()
model = load_model()
wordcloud = WordCloud(random_state = 8,
        normalize_plurals = False,
        width = 600, height= 300,
        max_words = 300,
        stopwords = [])

if __name__ == '__main__':
    run()
