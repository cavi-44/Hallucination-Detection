import re
from collections import Counter
import numpy as np
import json
import urllib.request
import zipfile
import os


def prepare_data(text, threshold=1e-4, table_size=1000000):
    #we get tokens using regular expression matching, all lowercase, only alphanumeric
    tokens = re.findall(r'\w+', text.lower())
    word_counts = Counter(tokens) #for each word - how many times it appeared
    total_words = len(tokens)

    vocab = list(word_counts.keys())
    vocab_size = len(vocab)
    word2idx = {word: i for i, word in enumerate(vocab)}  #mapping ids to words

    #subsampling frequent words (Paper #2)
    subsampled_corpus = []
    for word in tokens:
        freq = word_counts[word] / total_words
        p_discard = 1 - np.sqrt(threshold / freq) #the probability of discarding a word is calculated based on its frequency; the higher the frequency, the higher the probability of discard
        if np.random.rand() > p_discard:
            subsampled_corpus.append(word2idx[word])

    #unigram table (Paper #2)
    counts = np.array(list(word_counts.values()))
    p_unigram = counts ** 0.75 #flattens distribution (Paper #2)
    p_unigram /= np.sum(p_unigram)

    unigram_table = np.zeros(table_size, dtype=np.int32) #man, thats pretty pricey
    i = 0
    p_cumulative = p_unigram[0]
    for a in range(table_size):
        unigram_table[a] = i
        if a / table_size > p_cumulative: #inserting same word until it's presence exceeds it's probability
            i += 1
            if i >= vocab_size: i = vocab_size - 1
            p_cumulative += p_unigram[i]

    return subsampled_corpus, vocab_size, word2idx, unigram_table


def download_text8():
    url = "http://mattmahoney.net/dc/text8.zip"
    file_name = "text8.zip"

    if not os.path.exists("text8"):
        print("Downloading text8")
        urllib.request.urlretrieve(url, file_name)

        print("Unzipping")
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(".")

    with open("text8", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    print(f"Loaded {len(corpus_text.split())}")
    return corpus_text

def load_dataset(file_path):
    print(f"Loading dataset  {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
    print("Dataset loaded successfully.")
    return corpus_text
