import nltk
import pickle
from collections import Counter
from utils import save_obj

class Vocabulary(object):
    
    def __init__(self, captions_dict, vocab_threshold = 4,\
              start_word = '<BOS>', end_word = '<EOS>', unk_word = '<UNK>'):
        self.vocab_threshold = vocab_threshold # Minimum count of words for a word to be considered in the vocabulary
        self.start_word = start_word # Token to denote beginning of sentence, end of sentence and unknown word
        self.end_word = end_word
        self.unk_word = unk_word
        # Extract captions from captions_dict
        self.captions = []
        for sublist in captions_dict.values():
            for item in sublist:
                self.captions.append(item)
        # Initiate dictionaries to convert tokens to/from integers
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.get_vocab()
        self.save_vocab()
    
    def get_vocab(self):
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def add_captions(self):
        counter = Counter()
        for i, caption in enumerate(self.captions):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if i % 100 == 0:
                print(f"Tokenize captions: {i, len(self.captions)}")
        
        for word, count in counter.items():
            if count >= self.vocab_threshold:
                self.add_word(word)
    
    def save_vocab(self):
        save_obj(self, 'vocab')
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)