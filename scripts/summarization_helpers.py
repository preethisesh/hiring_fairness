import glob
import itertools
import os
import random
import re
import time
from collections import Counter

import evaluate
import numpy as np
import pandas as pd
from scipy import stats
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import textstat

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
regard = evaluate.load("regard")

def split_sentences_regex(text):
    return re.split(r'(?<=[.!?])\s+', text)
    
def get_readability_ease(text):
    """
    Returns Flesch Reading Ease score 
    More info: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
    """
    return textstat.flesch_reading_ease(text)

def get_reading_time(text):
    """
    Returns the reading time of the given text, assuming 14.69ms per character.
    """
    return textstat.reading_time(text)

def get_regard(text, split=True):
    """
    Returns regard score (regard aims to measure language polarity towards and social perceptions of a demographic)
    Only consider the positive regard category.
    """
    if split:
        return np.average([get_regard_single(t) for t in split_sentences_regex(text)])
    else:
        return get_regard_single(text)

def get_regard_single(text):
    return [x['score'] for x in regard.compute(data=[text])['regard'][0] if x['label'] == 'positive'][0]

def get_polarity(text, split=True):
    """
    Return polarity score between [-1.0, 1.0], where -1.0 is very negative and 1.0 is very positive.
    """
    if split:
        return np.average([get_polarity_single(t) for t in split_sentences_regex(text)])
    else:
        return get_polarity_single(text)  

def get_polarity_single(text):
    doc = nlp(text)
    return doc._.blob.polarity   
    
def get_subjectivity(text, split=True):
    """
    Returns subjectivity score between [0.0, 1.0], where 0.0 is very objective and 1.0 is very subjective.
    """
    if split:
        return np.average([get_subjectivity_single(t) for t in split_sentences_regex(text)])
    else:
        return get_subjectivity_single(text)
    
def get_subjectivity_single(text):
    doc = nlp(text)
    return doc._.blob.subjectivity

def compute_ttest(vals1, vals2):
    """
    Tests the null hypothesis that two related or repeated samples have identical expected values (measures invariance).
    """
    rel_stat = stats.ttest_rel(vals1, vals2, alternative='two-sided')
    return rel_stat