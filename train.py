#!/usr/bin/python
# -*- coding: cp1251 -*-

import string
import numpy as np
import random
import sys
import os
import argparse


def tokenize(text: str):
    text = text.lower()
    for punct in string.punctuation + string.digits + '–':
        text = text.replace(punct, '')
    tokens = text.split() 
    return tokens


def get_ngrams(n: int, tokens: list):
    ngrams = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return ngrams       


class NgramLanguageModel:

    def __init__(self):
        self.ngram_context = {}
        self.ngram_counter = {}

    def fit(self, tokens: list):
        for n in range(2, 5):
            ngrams = get_ngrams(n, tokens)
            for ngram in ngrams:
                if ngram in self.ngram_counter:
                    self.ngram_counter[ngram] += 1.0
                else:
                    self.ngram_counter[ngram] = 1.0

                context, candidate_token = ngram
                if context in self.ngram_context:
                    if candidate_token not in self.ngram_context[context]:
                        self.ngram_context[context].append(candidate_token)
                    else: 
                        pass
                else:
                    self.ngram_context[context] = [candidate_token]
        
    def token_probability(self, context, token):
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.ngram_context[context]))
            result = count_of_token / count_of_context
        except KeyError:
            result = 0.0
        return result

    def choose_next_token(self, context: tuple):
        tokens_probs = {}
        token_of_interest = self.ngram_context[context]
        for token in token_of_interest:
            tokens_probs[token] = self.token_probability(context, token)
        sm = sum(tokens_probs.values())
        token = np.random.choice(list(tokens_probs.keys()), p=[prob/sm for prob in list(tokens_probs.values())])
        return token
           
    def generate(self, length, prefix=''):
        if prefix == '':
            prefix = random.choice(list(self.ngram_context.keys()))
            prefix = ' '.join(prefix)
            print('Случайный префикс:', prefix)
        prefix_queue = tuple(tokenize(prefix))
        if prefix_queue not in self.ngram_context.keys():
            return print('Данного префикса в словаре нет')
        n = len(prefix_queue) + 1
        generated_text = [word for word in prefix_queue]
        for _ in range(length):
            next_word = self.choose_next_token(tuple(prefix_queue))
            generated_text.append(next_word)
            prefix_queue = tuple(reversed(generated_text[-1:-n:-1]))
        print(' '.join(generated_text))


def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--input-dir')
    parser.add_argument('--length',type=int, required=True)
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    tokens = []
    if args.input_dir is None:
        text = sys.stdin.read()
        tokens = tokenize(text)
    else:
        textFiles = []
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='cp1251') as f:
                    textFiles.append(f.read())
        for text in textFiles:
            tokens += tokenize(text)
    model = NgramLanguageModel()
    model.fit(tokens)
    model.generate(length=args.length, prefix=args.prefix)

if __name__ == '__main__':
    main()
    
