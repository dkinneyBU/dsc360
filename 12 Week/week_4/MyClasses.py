#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 08:03:56 2023

@author: david kinney
Classes for DSC360 assignments

"""
  
class Normalize_Corpus:
    
    def normalize(self, corpus, html_stripping=True, contraction_expansion=True, 
                  accented_char_removal=True, text_lower_case=True, 
                  text_lemmatization=True, special_char_removal=True, 
                  remove_digits=True, stopword_removal=True):
        import re
        from bs4 import BeautifulSoup
        import unicodedata
        from contractions import CONTRACTION_MAP
        import spacy
        import nltk
        from nltk.tokenize.toktok import ToktokTokenizer
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')
        nlp = spacy.load('en_core_web_sm')
        
        
        def strip_html_tags(text):
            soup = BeautifulSoup(text, "html.parser")
            [s.extract() for s in soup(['iframe', 'script'])]
            stripped_text = soup.get_text()
            stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
            return stripped_text
    
        def remove_accented_chars(text):
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            return text
        
        def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
            
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                              flags=re.IGNORECASE|re.DOTALL)
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match)\
                                        if contraction_mapping.get(match)\
                                        else contraction_mapping.get(match.lower())                       
                expanded_contraction = first_char+expanded_contraction[1:]
                return expanded_contraction
                
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text
    
        def lemmatize_text(text):
            text = nlp(text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        
        def remove_special_characters(text, remove_digits=False):
            pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
            text = re.sub(pattern, '', text)
            return text
        
        def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
            tokens = tokenizer.tokenize(text)
            tokens = [token.strip() for token in tokens]
            if is_lower_case:
                filtered_tokens = [token for token in tokens if token not in stopwords]
            else:
                filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
            filtered_text = ' '.join(filtered_tokens)    
            return filtered_text
    
        def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                             accented_char_removal=True, text_lower_case=True, 
                             text_lemmatization=True, special_char_removal=True, 
                             stopword_removal=True, remove_digits=True):
            
            # strip HTML
            print("Stripping HTML...")
            doc = corpus
            if html_stripping:
                # doc = strip_html_tags(doc)
                doc = doc.apply(lambda x: strip_html_tags(x))
            # remove accented characters
            print("Removing accented characters...")
            if accented_char_removal:
                # doc = remove_accented_chars(doc)
                doc = doc.apply(lambda x: remove_accented_chars(x))
            # expand contractions    
            print("Expnding contractions...")
            if contraction_expansion:
                # doc = expand_contractions(doc)
                doc = doc.apply(lambda x: expand_contractions(x))
            # lowercase the text   
            print("Converting to lowercase...")
            if text_lower_case:
                doc = doc.str.lower()
            # remove extra newlines
            print("Removing extra newlines...")
            doc = doc.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
            # lemmatize text
            print("Lemmatizing text...")
            if text_lemmatization:
                # doc = lemmatize_text(doc)
                doc = doc.apply(lambda x: lemmatize_text(x))
            # remove special characters and\or digits    
            print("Removing special characters and\or digits...")
            if special_char_removal:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                # doc = special_char_pattern.sub(" \\1 ", doc)
                # doc = remove_special_characters(doc, remove_digits=remove_digits)
                doc = doc.apply(lambda x: special_char_pattern.sub(" \\1 ", x))
                doc = doc.apply(lambda x: remove_special_characters(x, remove_digits=remove_digits))
            # remove extra whitespace
            print("Removing extra whitespace...")
            doc = doc.apply(lambda x: re.sub(' +', ' ', x))
            # remove stopwords
            print("Removing stopwords...")
            if stopword_removal:
                # doc = remove_stopwords(doc, is_lower_case=text_lower_case)
                doc = doc.apply(lambda x: remove_stopwords(x, is_lower_case=text_lower_case))
            # normalized_corpus.append(doc)
            normalized_corpus = doc
                
            return normalized_corpus
        
        # Weeks 4 and 5 require a string. Week 6 requires a series.
        # return normalize_corpus(corpus).to_string()
        return normalize_corpus(corpus)
    
    def BOW(norm_corpus):
        import pandas as pd
        
        print('\nBag of Words Model\n')
        # starting on page 208
        from sklearn.feature_extraction.text import CountVectorizer
        # get bag of words features in sparse format
        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(norm_corpus)
        # view non-zero feature positions in the sparse matrix
        # print(cv_matrix, '\n')
        
        # view dense representation
        # warning - might give a memory error if the data is too big
        cv_matrix = cv_matrix.toarray()
        # print(cv_matrix, '\n')
        
        # get all unique words in the corpus
        vocab = cv.get_feature_names_out()
        #show document feature vectors
        cv_df = pd.DataFrame(cv_matrix, columns=vocab)
        # print(cv_df, '\n')
        
        # you can set the n-gram range to 1,2 to get unigrams as well as bigrams
        bv = CountVectorizer(ngram_range=(2,2))
        bv_matrix = bv.fit_transform(norm_corpus)
        bv_matrix = bv_matrix.toarray()
        vocab = bv.get_feature_names_out()
        bv_df = pd.DataFrame(bv_matrix, columns=vocab)
        # print(bv_df, '\n')
        return bv_df
    
    def TfIdf_Transformer(norm_corpus):
        print('\nTf Idf Transformer:\n')
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        
        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(norm_corpus)

        tt = TfidfTransformer(norm = 'l2', use_idf=True)
        tt_matrix = tt.fit_transform(cv_matrix)
        tt_matrix = tt_matrix.toarray()
        vocab = cv.get_feature_names_out()
        return pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)
    
    def TfIdf_Vectorizer(norm_corpus):
        print('\nTf Idf Vectorizer:\n')
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
        tv_matrix = tv.fit_transform(norm_corpus)
        tv_matrix = tv_matrix.toarray()
        vocab = tv.get_feature_names_out()
        return tv_matrix, pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
    
    def Similarity_Matrix(tv_matrix):
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(tv_matrix)
        similarity_df = pd.DataFrame(similarity_matrix)
        return similarity_df