from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
import os

# Load necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer



# Load pre-trained model, words, classes, etc.
lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    intents = json.load(file)
app = Flask(__name__)
app.template_folder = 'templates'  # Set the template folder

# Chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Check if the user's message matches a specific intent
    if tag == 'your_custom_intent':
        # Add custom logic to generate a response
        result = 'Your custom response for this intent.'
    else:
        # Use the existing logic to select a response based on predicted intent
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    user_message = request.form['user_message']
    intents_list = predict_class(user_message)
    response = get_response(intents_list, intents)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)