import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

import numpy as np
import json, glob

import re, math, operator
from collections import Counter
import matplotlib
import numpy as np
from sklearn.preprocessing import normalize

import pickle


def unique_word_list(input_filename):
    input_file = open(input_filename, 'r')
    file_contents = input_file.read()
    input_file.close()
    word_list = file_contents.split()

    unique_list = []#file = open(output_filename, 'w')

    unique_words = set(word_list)

    for word in unique_words:
    
        unique_list.append(str(word))

    return unique_list

texts = []

for root, dirs, files in os.walk(r'./wiki_en'):
    for file in files:
        if file.endswith('.txt'):
            texts.append(file)

i = 0

while i<len(texts):

    path = "./wiki_en/" + texts[i]

    #print(unique_word_list(path))

    i = i + 1

i = 0

doc_data = []

while i<len(texts):

    path = "./wiki_en/" + texts[i]

    with open(path, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        doc_data.append(str(data))

    i = i+1

count_vectorizer = CountVectorizer(min_df=1)

with open("count_vectorizer.pkl", 'wb') as vectorizer_handle:
                    pickle.dump(count_vectorizer, vectorizer_handle)

term_freq_matrix = count_vectorizer.fit_transform(doc_data)
#print "Vocabulary:", count_vectorizer.vocabulary_

vocabulary = count_vectorizer.vocabulary_

with open("vocabulary.pkl", 'wb') as handle:
                    pickle.dump(vocabulary, handle)

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)

tf_idf_matrix = tfidf.transform(term_freq_matrix)

with open("tf_idf_matrix.pkl", 'wb') as tf_idf_matrix_handle:
                    pickle.dump(tf_idf_matrix, tf_idf_matrix_handle)


#print tf_idf_matrix.todense()


#print tf_idf_matrix.todense()

print "Enter a word: "
word = raw_input()

if vocabulary.has_key(word):

    word_index  = vocabulary[word]

    distance_from_other_words = {}
    vector_1 = tf_idf_matrix.todense()[:,word_index]
    for i in range(tf_idf_matrix.todense().shape[0]):

        #print(" Beginning processing.")
        if i==word_index:
            pass
        else:
            vector_2 = tf_idf_matrix.todense()[:,i]
            sim_score = cosine(vector_1, vector_2)
            distance_from_other_words[i] = sim_score

            #print("checking else part in processing.")

else:

    print "word not found in corpus. Cannot continue."


if vocabulary.has_key(word):

    print distance_from_other_words


#dict_keys = vocabulary.iterkeys()
#dict_vals = vocabulary.itervalues()


result_list = []

if vocabulary.has_key(word):

    pass

    #print("going into last if.")

    #dist_from_words_pointer1 = distance_from_other_words.iterkeys() #has the indices of the similar words
    #dist_from_words_pointer2 = distance_from_other_words.itervalues() #has the cosine similarities of the similar words

    #for vocab_index in vocabulary.itervalues():


        #for vocab_keys in vocabulary.iterkeys():

            #for sim_word_index in distance_from_other_words.iterkeys():

                #for sim_word_cosines in distance_from_other_words.itervalues():
                   
                    #if vocab_index == sim_word_index:
         
                        #print(vocab_keys," ",sim_word_cosines)
    
    #for i in range(len(distance_from_other_words)):

        #print("going into last first for.")

        #dict_keys = vocabulary.iterkeys()
        #dict_vals = vocabulary.itervalues()

        #for j in range(len(vocabulary)):

            #print("going into last second for.")


            #for vocab_index in vocabulary.itervalues():


             #   for vocab_keys in vocabulary.iterkeys():

              #      for sim_word_index in distance_from_other_words.iterkeys():


               #         for sim_word_cosines in distance_from_other_words.itervalues():


                   
                #            if vocab_index == sim_word_index:

            #if vocabulary[dict_vals] == dist_from_words_pointer1:
         
                 #               print(vocab_keys," ",sim_word_cosines)

            #dict_vals.next()
            #dict_keys.next()

        #dist_from_words_pointer1.next()
        #dist_from_words_pointer2.next()
            


        

