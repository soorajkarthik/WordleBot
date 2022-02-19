import itertools
import os

import numpy as np


DATA_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data"
)

POSSIBLE_WORD_LIST_FILE = os.path.join(DATA_DIRECTORY, "possible_words.txt")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIRECTORY, "pattern_matrix.npy")

EXACT_MATCH = np.uint8(2)
ALMOST_MATCH = np.uint8(1)
NO_MATCH = np.uint8(0)

PATTERN_MATRIX = None
WORD_INDEX_MAP = None


def get_usable_words(): 
    """
    Gets all of the possible answers for our Wordle bot

    returns: a list of str with the possible words
    """
    with open(POSSIBLE_WORD_LIST_FILE) as file:
        return [line.strip() for line in file.readlines()]


def save_pattern_matrix():
    """
    Saves creates a numpy file of a matrix where matrix[a, b] gives
    us the pattern if "a" was the guess and "b" was the answer. It
    represents this value as an integer. If we write this number in
    ternary, it is the pattern where 2 is a green match, 1 is a yellow, 
    and 0 is grey.
    """

    words = np.array([[ord(char) for char in word] for word in get_usable_words()])

    word_len = len(words[0])
    num_words = len(words)

    # per_letter_equality[a, b, i, j] = True if words[a][i] = words[b][j]
    per_letter_equality = np.zeros((num_words, num_words, word_len, word_len), dtype=bool)

    # pattern_matrix[a, b, i] = 0 | 1 | 2 (based on grey, yellow, green)
    pattern_matrix = np.zeros((num_words, num_words, word_len), dtype=np.uint8)

    for i1, i2 in itertools.product(range(word_len), range(word_len)):
        per_letter_equality[:, :, i1, i2] = np.equal.outer(words[:, i1], words[:, i2])

    for i in range(word_len):
        exact_matches = per_letter_equality[:, :, i, i].flatten()
        pattern_matrix[:, :, i].flat[exact_matches] = EXACT_MATCH
        
        for k in range(word_len):
            per_letter_equality[:, :, i, k].flat[exact_matches] = False
            per_letter_equality[:, :, k, i].flat[exact_matches] = False

    
    for i, j in itertools.product(range(word_len), range(word_len)):
        matches_ij = per_letter_equality[:, :, i, j].flatten()
        pattern_matrix[:, :, i].flat[matches_ij] = ALMOST_MATCH

        for k in range(word_len):
            per_letter_equality[:, :, i, k].flat[matches_ij] = False
            per_letter_equality[:, :, k, j].flat[matches_ij] = False

    # Dot product, gives us int representation of pattern, if we look at pattern in ternary
    pattern_matrix = np.dot(pattern_matrix, 3**np.arange(word_len, dtype=np.uint8))    

    np.save(PATTERN_MATRIX_FILE, pattern_matrix)


def get_pattern_matrix(words1, words2):
    """
    Returns a len(words1) x len(words2) matrix where matrix[a, b]
    gives us the pattern if "a" was a guess and "b" was the answer.
    It we get the value for every pair of words between words1 and
    words2. 

    """

    global PATTERN_MATRIX
    global WORD_INDEX_MAP

    if PATTERN_MATRIX is None or WORD_INDEX_MAP is None:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            save_pattern_matrix()

        PATTERN_MATRIX = np.load(PATTERN_MATRIX_FILE)
        WORD_INDEX_MAP = {word: index for (index, word) in enumerate(get_usable_words())}

    word1_indices = [WORD_INDEX_MAP[word] for word in words1] # Rows from pattern matrix
    word2_indices = [WORD_INDEX_MAP[word] for word in words2] # Colums from pattern matrix

    matrix_rows_cols = np.ix_(word1_indices, word2_indices)

    return PATTERN_MATRIX[matrix_rows_cols]
        

def calculate_expected_entropy(patterns):
    """
    Calculates the expected information of the guess given
    the list of patterns that the guess could potentially
    result in.
    """

    total_patterns = len(patterns)
    _, counts = np.unique(patterns, return_counts=True)
    probabilities = counts / total_patterns
    
    expected_entropy = np.sum(np.multiply(probabilities, -1*np.log2(probabilities)))

    return expected_entropy


def make_guess(words):
    """
    Makes a guess just based on the simple heuristic of 
    maximizing the entropy of our guess.
    """
    
    pattern_matrix = get_pattern_matrix(words, words)
    expected_entropies = np.apply_along_axis(calculate_expected_entropy, 1, pattern_matrix)
    guess = words[np.argmax(expected_entropies)]

    return guess


def trim_word_list(words, guess, guess_pattern):
    """
    Trim the list of possible words based on our guess
    and the pattern that we got from our guess.
    """
    
    all_word_patterns = get_pattern_matrix([guess], words).flatten()
    pattern_match_indices = np.where(all_word_patterns == guess_pattern)
    trimmed_words = words[pattern_match_indices]

    return trimmed_words 

def play_game():
    words = np.array(get_usable_words())
    
    while(len(words) > 0):
        guess = make_guess(words)
        print(guess)
        pattern_str = input("What was the pattern: ")[::-1]
        pattern = int(pattern_str, 3)
        words = trim_word_list(words, guess, pattern)

if __name__ == "__main__":
    play_game()