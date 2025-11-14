import torch
from torch.utils.data import Dataset
import re
from collections import Counter

# --- Configuration ---
FILE_PATH = '../dataset/Pride_and_Prejudice-Jane_Austen.txt'
# We must find the *actual* start and end of the novel's text
START_MARKER = "CHAPTER I."
END_MARKER = "***END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***"


def load_data(filepath):
    """Loads the full text from the file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        full_text = f.read()
    return full_text

def preprocess_text(text, start_marker, end_marker):
    # (Same as before, but with corrected markers)
    try:
        start_index = text.index(start_marker)
        end_index = text.index(end_marker)
        novel_text = text[start_index:end_index]
    except ValueError:
        print("Error: Start or end marker not found. Using full text.")
        novel_text = text

    novel_text = novel_text.lower()
    novel_text = re.sub(r'_([^_]+)_', r'\1', novel_text) # Handles _italics_ and _her_
    novel_text = re.sub(r'm\^{rs}', 'mrs', novel_text)
    novel_text = re.sub(r'[\{\}\^\*]', '', novel_text) # Removed * as well
    
    # --- Better Tokenization ---
    # This regex splits on spaces AND punctuation, keeping punctuation as its own token.
    # 'Mr. Bennet' becomes ['mr', '.', 'bennet']
    # This is a much more robust tokenizer than .split(' ')
    words = re.findall(r"[\w']+|[.,!?;]", novel_text)
    
    return words # Return the list of tokens

def build_vocab(all_tokens):
    # (Corrected logic from Critique)
    word_counts = Counter(all_tokens)
    sorted_vocab = [word for word, count in word_counts.most_common()]
    
    word_to_idx = {}
    idx_to_word = {}
    
    special_tokens = ['<PAD>', '<UNK>']
    for i, token in enumerate(special_tokens):
        word_to_idx[token] = i
        idx_to_word[i] = token
        
    for i, word in enumerate(sorted_vocab):
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            
    return word_to_idx, idx_to_word

# --- Main execution ---
raw_text = load_data(FILE_PATH)
clean_text = preprocess_text(raw_text, START_MARKER, END_MARKER)

print("--- Sample of Cleaned Text ---")
print(clean_text[:500])
print("\n...\n")

word_to_idx, idx_to_word = build_vocab(clean_text)

print(f"--- Vocabulary Stats ---")
print(f"Total words (tokens) in corpus: {len(clean_text)}")
print(f"Unique words (vocabulary size): {len(word_to_idx)}")

print("\n--- Sample of Mappings ---")
print(f"'elizabeth' -> {word_to_idx.get('elizabeth', 'Not found')}")
print(f"'darcy' -> {word_to_idx.get('darcy', 'Not found')}")
print(f"Index 1 ('<UNK>') -> {idx_to_word.get(1, 'Not found')}")

class TextDataset(Dataset):
    """
    A PyTorch Dataset for our Language Model.
    This class will take the full list of tokens and a sequence length,
    and prepare (input_sequence, target_word) pairs.
    """
    def __init__(self, all_tokens, word_to_idx, sequence_length=30):
        self.sequence_length = sequence_length
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.unknown_idx = word_to_idx['<UNK>']
        
        # 1. Convert all tokens to their corresponding indices
        #    Hint: Use .get(word, self.unknown_idx) to handle rare words
        #    that you might want to exclude from the vocab later.
        self.token_indices = [self.word_to_idx.get(word, self.unknown_idx) for word in all_tokens]

        # 2. Create the sequences
        #    self.sequences will be a list of (input_indices, target_index) tuples
        self.sequences = []
        
        # --- YOUR CODE GOES HERE ---
        # Slide a window across self.token_indices
        # Stop before you run out of words for a full sequence + target
        for i in range(len(self.token_indices) - sequence_length):
            # Input is a sequence of length `sequence_length`
            input_seq = self.token_indices[i : i + self.sequence_length]
            
            # Target is the single word *after* the input sequence
            target_word = self.token_indices[i + self.sequence_length]
            
            self.sequences.append((input_seq, target_word))
        # --- END OF YOUR CODE ---

    def __len__(self):
        """ This should return the total number of sequences we created. """
        # --- YOUR CODE GOES HERE ---
        return len(self.sequences)
        # --- END OF YOUR CODE ---

    def __getitem__(self, idx):
        """
        This should return one sample: the input and target,
        converted to PyTorch Tensors.
        """
        # --- YOUR CODE GOES HERE ---
        input_seq, target_word = self.sequences[idx]
        
        # Convert to tensors
        # .long() is important because these are indices, not continuous values
        input_tensor = torch.tensor(input_seq).long()
        target_tensor = torch.tensor(target_word).long()
        
        return input_tensor, target_tensor
        # --- END OF YOUR CODE ---

# --- Main execution (for testing) ---
if __name__ == "__main__":
    
    # 1. Load and process data
    raw_text = load_data(FILE_PATH)
    all_tokens = preprocess_text(raw_text, START_MARKER, END_MARKER)
    word_to_idx, idx_to_word = build_vocab(all_tokens)
    
    print(f"--- Vocabulary Stats ---")
    print(f"Total tokens in corpus: {len(all_tokens)}")
    print(f"Unique words (vocabulary size): {len(word_to_idx)}")
    
    # 2. Create the Dataset
    SEQ_LENGTH = 10 # Using a small sequence for testing
    dataset = TextDataset(all_tokens, word_to_idx, sequence_length=SEQ_LENGTH)
    
    print(f"\n--- Dataset Stats ---")
    print(f"Total sequences created: {len(dataset)}")
    
    # 3. Test the Dataset
    # This pulls the *first* sample from the dataset
    if len(dataset) > 0:
        first_input, first_target = dataset[0] 
        
        print(f"\n--- First Sample ---")
        print(f"Input indices: {first_input}")
        print(f"Target index: {first_target}")
        
        # Convert back to words to check
        input_words = [idx_to_word[idx.item()] for idx in first_input]
        target_word = idx_to_word[first_target.item()]
        
        print(f"Input as text: {' '.join(input_words)}")
        print(f"Target as text: {target_word}")
    
    # Check the 100th sample
    if len(dataset) > 100:
        input_100, target_100 = dataset[100]
        input_words_100 = ' '.join([idx_to_word[idx.item()] for idx in input_100])
        target_word_100 = idx_to_word[target_100.item()]
        
        print(f"\n--- 100th Sample ---")
        print(f"Input as text: {input_words_100}")
        print(f"Target as text: {target_word_100}")