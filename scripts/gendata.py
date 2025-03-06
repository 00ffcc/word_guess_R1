import datasets

def gen_gen(words, split):
    def gen():
        for idx, word in enumerate(words):
            # 转大写
            word = word.upper()
            word_length = len(word)
            prompt = f"""
    You are playing a word-guessing game.  Your goal is to guess a secret word of length {word_length} with the fewest possible queries. You can make guesses by using the following format:

    <query>YOUR_GUESS</query>

    Immediately following your query, you will receive a response in this format:

    <res>RESPONSE_STRING</res>

    The RESPONSE_STRING will be a string of the same length as the secret word, composed of the digits '0', '1', and '2'.  These digits represent the following:

    *   **0:** The letter in your guess at that position is correct (matches the secret word).
    *   **1:** The letter in your guess at that position is incorrect, and that letter does not appear anywhere in the secret word.
    *   **2:** The letter in your guess at that position is incorrect, but that letter *does* appear somewhere else in the secret word.

    Your objective is to guess the secret word in the fewest possible queries.  Use the information from each response to refine your subsequent guesses. Think step-by-step and explain your reasoning before each query. All words (both your guesses and the secret word) are in UPPERCASE letters.

    **Example (for a 5-letter word):**

    Let's assume, for the purpose of this example (and *only* this example), that the secret word is "SHAPE". You don't know this.
    <query>PLANE</query><res>21010</res>
    Reasoning: 'A' and 'E' are correct, and 'P' is in the word but in the wrong spot. 'L' and 'N' are not in the word. Let's test the position of 'P'.
    <query>SHAPE</query><res>00000</res>

    Now, begin playing. The secret word has {word_length} letters. Start guessing.

    """
            print(prompt)
            data = {
                    "data_source": "NLTK",
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {
                            "target": word,
                        }
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            yield data
    return gen

import random
import nltk
from nltk.corpus import words

def get_unique_english_words(num_words, lens):
    try:
        nltk.data.find('corpora/words') # Check if the words corpus is downloaded
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download('words')


    all_words = words.words()
    filtered_words = [word.lower() for word in all_words if len(word) in lens and word.isalpha()]
    unique_words = set(filtered_words)
    
    return random.sample(list(unique_words), num_words)

if __name__ == '__main__':
    lens = [3, 4, 5]
    file_name = "".join([str(l) for l in lens])
    train_num_words = 1024
    valid_num_words = 128

    all_words = get_unique_english_words(train_num_words + valid_num_words, lens)
    train_words = all_words[:train_num_words]
    valid_words = all_words[train_num_words:]

    train_dataset = datasets.Dataset.from_generator(gen_gen(train_words, 'train'))
    valid_dataset = datasets.Dataset.from_generator(gen_gen(valid_words, 'valid'))

    # save as parquet
    train_dataset.to_parquet(f"../data/word_guessing_{file_name}_train.parquet")
    valid_dataset.to_parquet(f"../data/word_guessing_{file_name}_valid.parquet")
