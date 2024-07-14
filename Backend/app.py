from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import unicodedata
import csv
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# This should contain your translations
translation_dict = {}

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def add_to_translation_dict(pair):
    english_sentence, yoruba_sentence = pair
    normalized_en_sentence = normalize_string(english_sentence)
    normalized_yo_sentence = normalize_string(yoruba_sentence)

    # Add sentence-level translation
    translation_dict[normalized_en_sentence] = normalized_yo_sentence

    english_words = normalized_en_sentence.split()
    yoruba_words = normalized_yo_sentence.split()

    for en_word, yo_word in zip(english_words, yoruba_words):
        translation_dict[en_word] = yo_word

def readLangs(lang1, lang2, reverse=False):
    datasets = ['data/eng-yor.csv']
    for filename in datasets:
        with open(filename, encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            pairs = [[s for s in row] for row in reader if len(row) == 2]
            for pair in pairs:
                add_to_translation_dict(pair)
    logging.debug(f"Translation dictionary loaded with {len(translation_dict)} entries.")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    english_text = data['text']
    normalized_text = normalize_string(english_text)

    logging.debug(f"Normalized text: {normalized_text}")

    # Check if the entire sentence exists in the dictionary
    if normalized_text in translation_dict:
        translated_text = translation_dict[normalized_text]
    else:
        # Translate word by word
        words = normalized_text.split()
        translated_words = []
        all_words_translated = True  # Flag to check if all words are translated

        for word in words:
            translation = translation_dict.get(word, None)
            if translation:
                translated_words.append(translation)
            else:
                translated_words.append(word)  # If not found, use the original word
                all_words_translated = False
        
        translated_text = ' '.join(translated_words)
        
        if not all_words_translated:
            logging.warning(f"Translation not found for some words in: {english_text}")

    return jsonify({'translation': translated_text})

if __name__ == '__main__':
    readLangs('eng', 'yor')
    app.run(debug=True)
