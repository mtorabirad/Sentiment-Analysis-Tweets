import os
import pandas as pd
import spacy
import re
from joblib import Parallel, delayed
from spacy.lang.en.stop_words import STOP_WORDS

import en_core_web_sm
nlp = en_core_web_sm.load(disable=['tagger', 'parser', 'ner'])
stopwordfile = r'AfterDec28\stopwords.txt'


def _get_stopwords(from_file=False):
    "Return a set of stopwords read in from a file."
    if from_file:
        with open(stopwordfile) as f:
            stopwords = []
            for line in f:
                stopwords.append(line.strip("\n"))
    else:
        stopwords = STOP_WORDS
    # Convert to set for performance
    stopwords_set = set(stopwords)
    return stopwords_set


def _lemmatize_pipe(doc):
    stop_words = _get_stopwords(from_file=False)

    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stop_words]
    return lemma_list


def chunker(iterable, total_length, chunk):
    return (iterable[pos: pos + chunk] for pos in range(0, total_length,
                                                        chunk))


def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def _process_chunk(texts):
    # texts is a pandas series.
    preproc_pipe = []
    # Process the texts as a stream using nlp.pipe and buffer them in batches
    # instead of one-by-one. This is usually much more efficient.
    # nlp.pipe() is a generator that processes text as stream and yields
    # a doc object in order.
    # See https://spacy.io/api/language
    # There are two ways to get values from generators: the next() function
    # and a for loop. The for loop is often the preferred method.
    # What is lemmatize_pipe doing here?
    # batch_size: The number of texts to buffer. What does this mean?

    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(_lemmatize_pipe(doc))
    return preproc_pipe


def parallel_lemmatize_in_chunks(df, chunk, **parameters):
    # chunk_size = parameters['chunk_size']
    executor = Parallel(**parameters)
    do = delayed(_process_chunk)
    tasks = (do(chunk) for chunk in chunker(df['clean'], len(df),
                                            chunk=chunk))
    result = executor(tasks)
    return flatten(result)
