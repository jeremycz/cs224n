#%% PACKAGES
import collections
import nltk
import numpy as np
from tqdm import tqdm
import time

#%% CONSTANTS
VOCAB_SIZE = 100
WINDOW_SIZE = 1
EMBEDDING_SIZE = 100

assert EMBEDDING_SIZE <= VOCAB_SIZE, "Embedding size must be <= vocab size due to output dimensions from SVD"


#%%
with open("week_01/data/text8", "r") as f:
    data = f.read()

data = data.split()

print("Dataset size: {}".format(len(data)))

#%%
# Get x most common words to form vocabulary
word_list = collections.Counter(data).most_common(VOCAB_SIZE)

vocab = dict()
rev_vocab = dict()

# Set key/values for OOV token
vocab["OOV"] = 0
rev_vocab[0] = "OOV"

# Used to count words in dataset covered by vocabulary
word_sum = 0

for i, word in enumerate(word_list):
    vocab[word[0]] = i + 1
    rev_vocab[i + 1] = word[0]
    word_sum += word[1]

print("Vocabulary size {} covers {:.2f}% of dataset".format(VOCAB_SIZE, word_sum / len(data) * 100))

#%% Tokenize dataset
print("Tokenizing dataset")
tokenized_data = [vocab[word] if word in vocab else 0 for word in tqdm(data)]

#%% Form word co-occurence count matrix
X = np.zeros((VOCAB_SIZE + 1, VOCAB_SIZE + 1)) # VOCAB_SIZE + 1 to include OOV token

print("Generating co-occurence matrix")
for i, target_word in enumerate(tqdm(tokenized_data)):
    start_ind = max(0, i - WINDOW_SIZE)
    end_ind = min(len(tokenized_data), i + WINDOW_SIZE + 1)

    context_words = tokenized_data[start_ind:i] + tokenized_data[i:end_ind]

    # Faster than indexing using a list
    for context_word in context_words:
        X[target_word, context_word] += 1
    

#%% Run SVD
print("Performing SVD")
start_time = time.time()
U, S, V = np.linalg.svd(X, hermitian=True)
end_time = time.time()
print("Time taken for SVD: {%.3f}s".format(end_time - start_time))

# Truncate U matrix to get word vectors
variance_captured = np.sum(S[:EMBEDDING_SIZE]) / np.sum(S)

print("Variance captured by embedding dimension: {:.2f}%".format(variance_captured * 100))
