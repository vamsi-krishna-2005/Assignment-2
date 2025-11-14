import torch
from torch.utils.data import Dataset
import re
from collections import Counter

def load_data(filepath):
    """Loads the full text from the file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        full_text = f.read()
    return full_text

def preprocess_text(text, start_marker, end_marker):
    """Cleans and tokenizes the raw text."""
    try:
        start_index = text.index(start_marker)
        end_index = text.index(end_marker)
        novel_text = text[start_index:end_index]
    except ValueError:
        print("Error: Start or end marker not found. Using full text.")
        novel_text = text

    novel_text = novel_text.lower()
    novel_text = re.sub(r'_([^_]+)_', r'\1', novel_text)
    novel_text = re.sub(r'm\^{rs}', 'mrs', novel_text)
    novel_text = re.sub(r'[\{\}\^\*]', '', novel_text)
    words = re.findall(r"[\w']+|[.,!?;]", novel_text)
    return words

def build_vocab(all_tokens, vocab_size_limit):
    """Builds vocab mappings, pruned to a specific size."""
    word_counts = Counter(all_tokens)
    # Get the most common words, leaving space for <PAD> and <UNK>
    sorted_vocab = [word for word, count in word_counts.most_common(vocab_size_limit - 2)]
    
    word_to_idx = {}
    idx_to_word = {}
    
    # Add special tokens first
    special_tokens = ['<PAD>', '<UNK>']
    for i, token in enumerate(special_tokens):
        word_to_idx[token] = i
        idx_to_word[i] = token
        
    # Add the rest of the vocabulary
    for i, word in enumerate(sorted_vocab):
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            
    return word_to_idx, idx_to_word

class TextDataset(Dataset):
    """PyTorch Dataset for (sequence, target) pairs."""
    def __init__(self, all_tokens, word_to_idx, sequence_length=30):
        self.sequence_length = sequence_length
        self.word_to_idx = word_to_idx
        self.unknown_idx = word_to_idx['<UNK>']
        
        # Convert all tokens to indices
        self.token_indices = [self.word_to_idx.get(word, self.unknown_idx) for word in all_tokens]
        
        self.sequences = []
        # Create sequences
        for i in range(len(self.token_indices) - sequence_length):
            input_seq = self.token_indices[i : i + self.sequence_length]
            target_word = self.token_indices[i + self.sequence_length]
            self.sequences.append((input_seq, target_word))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        return torch.tensor(input_seq).long(), torch.tensor(target_word).long()