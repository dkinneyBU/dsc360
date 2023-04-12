#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 08:44:21 2023

Solutions for DSC 360 Week 4 assignment

@author: Professor David Kinney, Bellevue University, College of Science & Tech.

"""

# %% imports and globals

from pprint import pprint
import nltk
import numpy as np
import pandas as pd
import spacy
from MyClasses import Normalize_Corpus

nlp = spacy.load("en_core_web_sm")

# %%
df_big_text = pd.read_fwf("./Week 4/big.txt", header=None)
corpus = df_big_text[0]

# corpus.apply(lambda x: print(type(x)))

# %%

normalizer = Normalize_Corpus()
normalized = normalizer.normalize(corpus)

# %%
with open('big.txt', 'r') as f: 
    text = f.read().replace('\n', '').replace('\t', '')

sample = text[0:1021]    

# %% tokens
words = nltk.word_tokenize(sample)
print(np.array(words))

# %% lemmas

def lemmatize_text(text):
    nlp = spacy.load('en_core_web_sm')
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
        
print(lemmatize_text(sample))
        
# %% parts of speech
sample_nlp = nlp(sample)

# tag PoS with spacy
spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in sample_nlp]
# the .T in the book transposes rows and columsn, but it's harder to read
print("spaCy\n")
pprint(pd.DataFrame(spacy_pos_tagged, columns=['Word', 'POS tag', 'Tag type']))

# POS tagging with nltk
print('\n', 'NLTK')
import nltk
# only need the following two lines one time
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
nltk_pos_tagged = nltk.pos_tag(nltk.word_tokenize(sample), tagset='universal')
pprint(pd.DataFrame(nltk_pos_tagged, columns=['Word', 'POS tag']))

# %% dependencies
sentence_nlp = nlp(sample)
dependency_pattern = '{left}<---{word}[{w_type}]--->{right}\n--------'
for token in sentence_nlp:
    print(dependency_pattern.format(word=token.orth_, w_type=token.dep_,
                                    left=[t.orth_ for t in token.lefts],
                                    right=[t.orth_ for t in token.rights]))
                                             
from spacy import displacy
displacy.render(sentence_nlp, jupyter=True, style='dep',
                options={'distance': 100,
                        'arrow_stroke': 2,
                        'arrow_width': 8})
