import argparse
import os
import time

import keyboard
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from processing import get_usable_words, make_guess, trim_word_list

DRIVER_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "drivers"
)

CHROME_DRIVER_FILE = os.path.join(DRIVER_DIRECTORY, "chromedriver.exe")

WORDLE_URL = "https://www.powerlanguage.co.uk/wordle/"
WORDMASTER_URL = "https://octokatherine.github.io/word-master/"


def start_game_manual(is_classic_wordle=False):   
    """
    Start manual entering of wordle pattern results.
    """
    usable_words = get_usable_words(is_classic_wordle=is_classic_wordle)
    while(True):
        words = np.copy(usable_words)
        while(True):
            guess = make_guess(words, is_classic_wordle=is_classic_wordle)
            print(guess)
            pattern_str = input("What was the pattern: ")[::-1]
            pattern = int(pattern_str, 3)
            
            if pattern == 3**5 - 1:
                break

            words = trim_word_list(words, guess, pattern, is_classic_wordle=is_classic_wordle)


def play_game_automated(words, game_rows, driver=None, is_classic_wordle=False):
    """
    Plays wordle automatically using keyboard library. Make sure
    you are focused on the window opened up by selenium.
    """

    for guess_num in range(6):
        guess = make_guess(words, is_classic_wordle=is_classic_wordle)
        keyboard.write(guess, delay=0.05)
        keyboard.press_and_release('enter')

        pattern_str = []

        if is_classic_wordle:
            time.sleep(2) # need to sleep to wait for animation to finish
            
            row = driver.execute_script('return arguments[0].shadowRoot', game_rows[guess_num])
            tiles = row.find_elements(By.CSS_SELECTOR, "game-tile")
            evaluation_map = {
                "correct": "2",
                "present": "1",
                "absent": "0"
            }
            
            for tile in tiles:
                pattern_str.append(evaluation_map[str(tile.get_attribute("evaluation"))])
        else:
            for tile in game_rows[guess_num]:
                if 'nm-inset-n-green' in tile.get_attribute("class"):
                    pattern_str.append("2")
                elif 'nm-inset-yellow-500' in tile.get_attribute("class"):
                    pattern_str.append("1")
                elif 'nm-inset-n-gray' in tile.get_attribute("class"):
                    pattern_str.append("0")

        pattern = int("".join(pattern_str[::-1]), 3)

        if pattern == 3**5 - 1:
            return [guess]

        words = trim_word_list(words, guess, pattern, is_classic_wordle=is_classic_wordle)

    return words        

def start_game_automated(is_classic_wordle=False, num_rounds=100):
    """
    Opens browser with website based on classic or not. Then 
    starts playing wordle automatically.
    """

    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(CHROME_DRIVER_FILE), options=chrome_options)    

    words = get_usable_words(is_classic_wordle=is_classic_wordle)
    start_button = 'esc'

    if is_classic_wordle:
        driver.get(WORDLE_URL)
        keyboard.wait(start_button)
           
        game_app = driver.find_element(By.TAG_NAME, 'game-app')
        board = driver.execute_script("return arguments[0].shadowRoot.getElementById('board')", game_app)
        game_rows = board.find_elements(By.TAG_NAME, 'game-row')

        play_game_automated(np.copy(words), game_rows, driver=driver, is_classic_wordle=is_classic_wordle)
    else:
        driver.get(WORDMASTER_URL)
        keyboard.wait(start_button)

        for _ in range(num_rounds):
            game_rows = np.array(driver.find_elements(By.TAG_NAME, 'span')).reshape(6, 5)

            play_game_automated(np.copy(words), game_rows, is_classic_wordle=is_classic_wordle)
            
            time.sleep(2)
            keyboard.press_and_release('esc')
            driver.find_element(By.XPATH, '//button[text()="Play Again"]').click()
            time.sleep(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default=False)
    parser.add_argument('-c', default=False)
    parser.add_argument('-r', default=100)
    args = parser.parse_args()

    manual = bool(args.m)
    classic = bool(args.c)
    rounds = int(args.r)

    if manual:
        start_game_manual(is_classic_wordle=classic)
    else:
        start_game_automated(is_classic_wordle=classic, num_rounds=rounds)
