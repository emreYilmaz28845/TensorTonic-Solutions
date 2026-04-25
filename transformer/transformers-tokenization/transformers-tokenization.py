import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # Add special tokens first
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for token in special_tokens:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size+=1

        unique_words = set()

        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size+=1
                    
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        # start the encode with bos 
        ids = []

        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])

        
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for i in ids:
            word = self.id_to_word.get(i, self.unk_token)
                
            words.append(word)
            
            
        return " ".join(words)
