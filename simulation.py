import argparse
from random import random
from time import time

import numpy as np
from tqdm import tqdm

from processing import (get_pattern_matrix, get_usable_words, make_guess,
                        save_pattern_matrix, trim_word_list)


def run_simulation(all_words, words_to_guess, is_classic_wordle=False, method='max_entropy'):

    num_rounds = len(words_to_guess)
    total_guesses = 0
    num_correct = 0

    print("Presaving pattern matrix...\n")
    get_pattern_matrix(all_words, all_words, is_classic_wordle=is_classic_wordle)

    print("Pattern matrix saved, starting simulation...\n\n")

    start_time = time()

    for word_to_guess in tqdm(words_to_guess):
        words = np.copy(all_words)
        for i in range(6):
            guess = make_guess(words, is_classic_wordle=is_classic_wordle, method=method)
            pattern = get_pattern_matrix([guess], [word_to_guess], is_classic_wordle=is_classic_wordle)[0, 0]

            if pattern == 3**5 - 1:
                total_guesses += (i + 1)
                num_correct += 1
                break

            words = trim_word_list(words, guess, pattern, is_classic_wordle=is_classic_wordle)

    end_time = time()
    
    total_time = end_time - start_time
    avg_time_per_guess = total_time / num_rounds
    percent_correct = (num_correct / num_rounds) * 100
    avg_num_guesses = total_guesses / num_rounds

    print("\n\n=============================================================")
    print(f"Simulation completed")
    print(f"Rounds: {num_rounds}")
    print(f"Method: {method}")
    print(f"Total time: {total_time:.2f}")
    print(f"Average time per guess: {avg_time_per_guess:.2f}")
    print(f"Number correct: {num_correct}/{num_rounds} = {percent_correct:.2f}%")
    print(f"Average guesses per word:  {avg_num_guesses:.2f}")
    print("=============================================================\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--classic', action='store_true', help='Simulate wordle classic. Default is WordMaster.')
    parser.add_argument('--method', type=str, default='max_entropy', choices=['max_entropy', 'max_two_step_entropy'], 
        help='The heuristic used to make a guess. Default is chosing the word with the max expected entropy')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rounds', type=int, default=100, help='Number of words to simulate. Default is 100.')
    group.add_argument('--all_words', action='store_true', help='Run simulation on all words.')

    args = parser.parse_args()

    classic = args.classic
    method = args.method
    rounds = args.rounds
    all_words = args.all_words

    usable_words = get_usable_words(is_classic_wordle=classic)
    words_to_guess = np.copy(usable_words) if all_words else np.random.choice(usable_words, rounds)

    run_simulation(usable_words, words_to_guess, is_classic_wordle=classic, method=method)
