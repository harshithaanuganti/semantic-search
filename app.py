#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Harshitha Anuganti
"""

import streamlit as st
import pandas as pd
import pickle as pkl
from sentence_transformers import SentenceTransformer, util
import re

import plotly.express as px

st.title("Semantic Search")
st.markdown("MSBA 6490 | Assignment2")
st.markdown("This is v1.0")

@st.cache(persist=True)
def load_data():
    with open("df.pkl", "rb") as file1:
        df = pkl.load(file1)
    return df

def load_corpus():
    with open("corpus_embeddings.pkl", "rb") as file1:
        corpus_embeddings = pkl.load(file1)
    return corpus_embeddings

def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

def get_content(string):
    string_pattern = r"(?<=    )(.*)"
    # compile string pattern to re.Pattern object
    regex_pattern = re.compile(string_pattern)
    return str(regex_pattern.findall(string)[0])

def run():

    df=load_data()
    corpus_embeddings=load_corpus()
    model=load_model()

    query = st.text_area("Search Here:")
    query_embedding = model.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=st.slider('Maximum results to be displayed:', 0, 100, 25))
    hits = hits[0]  # Get the hits for the first query

    for hit in hits:
        row_dict = df.loc[df['all_review'] == corpus[hit['corpus_id']]]
        print("\nHotel:  ", get_content(str(row_dict['hotelName'])))
        print(get_content(str(row_dict['sum_review'])), "\n")







if __name__ == '__main__':
    run()
