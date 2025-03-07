import datasets


def gen_gen(words, split):
    def gen():
        for idx, word in enumerate(words):
            # 转大写
            word = word.upper()
            word_length = len(word)
            prompt = f"""
Let's play Wordle. Your goal is to guess the secret {word_length}-letter word. You can query a word using the following format:

<query>YOURWORD</query>

Where 'YOURWORD' is the {word_length}-letter word you want to guess. The response will immediately follow your query and will be in this format:

<response>RESPONSE</response>

Where 'RESPONSE' describes the result of your guess using a natural-language style explanation for each letter.

For example, if the secret word is "SHAPE" and you query "CRANE", the query would be:

<query>CRANE</query>

the response would be:

<response>The first letter, C, is not in the word. The second letter, R, is in the word but in the wrong position. The third letter, A, is in the word but in the wrong position. The fourth letter, N, is not in the word. The fifth letter, E, is in the correct position.</response>

You should make your queries one at a time, and only query one word at a time. It should only contains the word itself, no other text.
Your objective is to guess the secret word in as few queries as possible.
    """
            print(prompt)
            data = {
                    "data_source": "gen",
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
    with open("google-10000-english-no-swears.txt", 'r', encoding='utf-8') as f:
        all_words = [line.strip() for line in f]
    filtered_words = [word.upper() for word in all_words if len(word) in lens and word.isalpha()]
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
