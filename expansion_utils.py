import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os, sys
import csv
import fasttext
import fasttext.util
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import expansion_utils as utils

# input: file of English emotions, output filename, output language to translate to
# output: none; creates 1:1 translation file with specified name 
def translate_emotions(source_file, output_file, dest_lang):
    emotion_string = ""
    with open(source_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            emotion_string += ("I feel " + row[0].lower() + '\n')
    translator = Translator()
    translator.raise_Exception = True
    translated_emotions = translator.translate(emotion_string, src='en', dest=dest_lang).text
    with open(output_file, "w") as csvfile:
        csvfile.write(translated_emotions)

# input: fasttext language
# output: dictionary of all embeddings in the form {word: embedding} + dictionary of frequencies
def load_fasttext(language):
    if language == 'English':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.en.300.bin'
    elif language == 'Hindi':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.hi.300.bin'
    elif language == 'Chinese':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.zh.300.bin'
    elif language == 'Spanish':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.es.300.bin'
    elif language == 'Japanese':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.ja.300.bin'
    else:
        raise ValueError('No pre-trained model for language: {}'.format(language))
    
    ft = fasttext.load_model(pre_trained_model)
    freqs = ft.get_words(include_freq=True, on_unicode_error='replace')
    return ft, freqs

# input: list of seed words, full fasttext embeddings
# output: list of corresponding seed word embeddings
def get_emb_ls(seed_word_ls, embeddings):
    seed_emb_ls = []
    for seed in seed_word_ls:
        seed_emb = np.mean([embeddings[x] for x in seed.split(' ') if x in embeddings], axis = 0)
        seed_emb_ls.append(seed_emb)
    return seed_emb_ls

# input: list of embeddings
# output: center embedding
def get_center_emb(emb_ls):
    return np.mean(emb_ls, axis = 0)

# input: center embedding, output size, full fasttest embeddings
# output: list of all words with cosine similarity to the center, descending order
def sort_words(seed_center_emb, output_size, embeddings):
    final_list = []
    min_sim = 0 
    all_words = list(embeddings.keys())
    for word in tqdm(all_words):
        cos_sim = cosine_similarity([seed_center_emb],[embeddings[word]])[0][0]
        if cos_sim > min_sim:
            if len(final_list) == output_size:
                final_list = final_list[:-1]
            final_list.append((word,cos_sim))
            final_list.sort(key = lambda x: x[1], reverse = True)
            min_sim = final_list[-1][1]
  
    return final_list

# input: list of seed words, output size, full fasttext embeddings
# output: list of all words with cosine similarity to the seed centroid, descending order
def full_expansion(seed_words, output_size, embeddings):
    seed_emb_ls = get_emb_ls(seed_words, embeddings)
    seed_center_emb = get_center_emb(seed_emb_ls)
    lexica = sort_words(seed_center_emb, output_size, embeddings)
    return lexica

# input: list of seed words, output size, full fasttext embeddings
# output: dictionary of all words with cosine similarity to each individual word, descending order
def individual_expansion(seed_words, output_size, embeddings):
    full_lexica = {}
    for seed_word in seed_words:
        seed_emb = get_emb_ls([seed_word], embeddings)[0]
        lexica = sort_words(seed_emb, output_size, embeddings)
        full_lexica[seed_word] = lexica
    return lexica

#input: list of words to purify, list of fasttext frequencies
#output: purified list with
def purify_list(nearest_words, freqs):
    final_lists = {}
    for emotion in nearest_words:
        words = nearest_words[emotion]
        emotion_list_curr = []
        for word in words:
            try:
                freq = freqs[(word[0]).replace("-", " ")]
            except:
                freq = 0
            if(freq > 100000):
                emotion_list_curr.append(word)
        final_lists[emotion] = emotion_list_curr
    return final_lists


# input: list of seed words, output size, full fasttext embeddings
# output: list of all words with cosine similarity to the seed centroid, descending order
def emotion_expansion(emotion, model, source_file):
    seed_emb = model[emotion]
    lexica = sort_emotions(seed_emb, model, source_file)
    return lexica

# input: center embedding, output size, full fasttest embeddings, file containing list of emotions
# output: list of all words with cosine similarity to the center, descending order
def sort_emotions(seed_center_emb, model, source_file):    
    all_emotions = []
    with open(source_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            all_emotions.append(row[0])
   
    final_list = []
    for word in tqdm(all_emotions):
        try:
            cos_sim = cosine_similarity([seed_center_emb],[model[word]])[0][0]
        except KeyError:
            continue
        final_list.append((word,cos_sim))    
    final_list.sort(key = lambda x: x[1], reverse = True)
    return final_list

def get_emotion_dict(source_file):
    emotion_dict = []
    with open(source_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            emotion_dict.append(row[0])
    return emotion_dict