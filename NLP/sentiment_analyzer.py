import os
import pandas as pd
import spacy
import re
from joblib import Parallel, delayed
from spacy.lang.en.stop_words import STOP_WORDS
from NLP.lemmatizer import flatten
from NLP.lemmatizer import chunker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import en_core_web_sm
nlp = en_core_web_sm.load(disable=['tagger', 'parser', 'ner'])


# The next step is to vectorize.
# textcat: a built-in pipeline component that spaCy provides to assign
# categories (or labels) to text data
# The next step is to extract features representing the sentiment of a tweet.

analyser = SentimentIntensityAnalyzer()


def _sentiment_pipe(doc):
    sentiment_list = analyser.polarity_scores(str(doc))
    return sentiment_list


def _sentiment_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(_sentiment_pipe(doc))
    return preproc_pipe


def parallel_sentiment_in_chunks(df, chunk, **para):
    # chunk_size = para['chunk_size']
    # executor = Parallel(n_jobs=para['n_jobs'], backend=para['backend'],
    #                     prefer=para["prefer"])
    executor = Parallel(**para)

    do = delayed(_sentiment_chunk)
    tasks = (do(chunk) for chunk in chunker(df['clean'], len(df),
                                            chunk=chunk))
    result = executor(tasks)
    return flatten(result)
