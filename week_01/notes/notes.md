# Week 1 Notes

## Introduction

- Language is a system constructed to convey meaning
- Words are *signifiers* which map to a *signified* (idea/thing)

### Examples of tasks

Goal: Design algorithms to allow computers to "understand" natural language in order to perform some task.

## Word Representations

- How should words be represented in order to be used as input to models? 
- Earlier NLP work treated words as atomic symbols.
  - Denotational semantics - concept of representing an idea as a symbol - localist representation
- To perform well on most NLP tasks we need some notion of similarity/difference between words (as measured using distance measures such as Jaccard, Cosine, Euclidean etc.)

## Word Vectors

- Estimated 13m tokens for the English language
- Goal is to encode word tokens into a vector that represents some point in "word" space.
- The idea is that maybe there exists some $N$-dimensional space (where $N \ll 13\times 10^6$) that is sufficient to encode all semantics of our language.

## SVD Methods

- Loop over dataset and get word co-occurrence counts in a matrix $X$, then perform SVD on $X$ to obtain $USV^T$ where the rows of $U$ are the word embeddings for all the words in the dictionary.
- SVD methods do not scale well for big matrices

Methods for generating $X$:
- Word-document matrix: Increment $X_{ij}$ for ever occurence of word $i$ in document $j$
  - Resulting matrix is very large, and scales with the number of documents
- Window based co-occurrence: Count the number of times each word appears inside a window of a particular size around the word of interest.

## Iteration Based Methods - Word2Vec

$$
x=1
$$