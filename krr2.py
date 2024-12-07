import streamlit as st
import pandas as pd
import numpy as np
import nltk
import language_tool_python
import spacy
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBRegressor
import re
import math

# Streamlit App

# Title and description
st.title('Automated Essay Scoring')
st.write("""
    This web app uses machine learning to score essays based on grammar, structure, vocabulary, and more.
    Upload your essay file (CSV format) below to get the score.
""")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    # Read the uploaded CSV
    df2 = pd.read_csv(uploaded_file)

    # Preprocessing and feature engineering
    df2.drop(['TEXT'], axis=1, inplace=True)
    df2['SCORE'] = df2['cEXT'] + df2['cNEU'] + df2['cAGR'] + df2['cCON'] + df2['cOPN']

    df2['EMBEDDINGS'] = [np.fromstring(i.strip('[]'), sep=' ') for i in df2['EMBEDDINGS']]
    df2['embedding_norm'] = df2['EMBEDDINGS'].apply(lambda x: np.linalg.norm(x))

    # Outlier removal
    Q1 = df2['embedding_norm'].quantile(0.15)
    Q3 = df2['embedding_norm'].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df2 = df2[(df2['embedding_norm'] >= lower_bound) & (df2['embedding_norm'] <= upper_bound)]
    df2 = df2.drop(columns=['embedding_norm'])

    x = df2['EMBEDDINGS'].apply(pd.Series)
    y = df2['SCORE']

    # Resampling
    ros = RandomOverSampler(random_state=42)
    x, y = ros.fit_resample(x, y)

    # Scaling
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # XGBoost model
    xgb_regressor = XGBRegressor(n_estimators=500, learning_rate=0.2)
    xgb_regressor.fit(x, y)

    # Show message after processing
    st.write("File processed successfully!")

    # Function to evaluate essay quality
    def grammar_score(text):
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        error_count = len(matches)
        score = max(0, 1 - error_count / 10)
        return score

    def structure_score(text):
        blob = TextBlob(text)
        sentence_lengths = [len(sentence.split()) for sentence in blob.sentences]
        if len(sentence_lengths) < 6:
            return 0.2
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        if avg_sentence_length < 10:
            score = avg_sentence_length / 10
        elif avg_sentence_length > 20:
            score = max(0, 1 - (avg_sentence_length - 20) / 10)
        else:
            score = 1
        return round(score, 2)

    def flow_score(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        transitions = {"however", "therefore", "moreover", "thus", "although", "meanwhile"}
        count_transitions = sum(1 for token in doc if token.text.lower() in transitions)
        score = min(5, count_transitions)
        score = score / 5
        return score

    def word_count(text, ideal_word_count=300, tolerance=50):
        word_count = len(text.split())
        deviation = abs(word_count - ideal_word_count)
        if deviation <= tolerance:
            return 1
        elif deviation <= tolerance * 2:
            return 0.8
        elif deviation <= tolerance * 3:
            return 0.6
        elif deviation <= tolerance * 4:
            return 0.4
        else:
            return 0.2

    def vocabulary_score(text):
        words = text.split()
        unique_words = set(words)
        vocab_richness = len(unique_words) / len(words) if words else 0
        score = min(1, vocab_richness * 3)
        return score

    def evaluate_score(text):
        grammar = grammar_score(text)
        structure = structure_score(text)
        flow = flow_score(text)
        count = word_count(text)
        vocabulary = vocabulary_score(text)
        total = grammar + structure + flow + count + vocabulary
        return total

    def embed(text):
        at = AutoTokenizer.from_pretrained('bert-base-uncased')
        am = AutoModel.from_pretrained('bert-base-uncased')
        in1 = at(text, return_tensors="pt", truncation=True, padding=True)
        outputs = am(**in1)
        sem = torch.mean(outputs.last_hidden_state, dim=1)
        return sem.squeeze().detach().numpy()

    # User's essay input
    essay = st.text_area("Paste your essay here:", value="", height=300)

    if essay:
        # Preprocessing the essay
        essay_cleaned = clean(essay)
        essay_corrected = correction(essay_cleaned)
        essay_final = correct(essay_corrected)

        # Embedding
        embeddings = embed(essay_final)
        embeddings = sc.transform(embeddings.reshape(1, -1))

        # Prediction score
        sc1 = evaluate_score(essay)
        sc2 = xgb_regressor.predict(embeddings)[0]

        if sc2 < 0:
            sc2 = 0
        if sc2 > 5:
            sc2 = 5

        final_score = math.ceil(sc1 + sc2)

        st.write(f"**Final Score:** {final_score}")
