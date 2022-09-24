import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime

nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')


data_file = open('./intents.json')
data = json.load(data_file)

words = []
classes = []
data_X = []
data_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_X.append(pattern)
        data_y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

model = tf.keras.models.load_model("./checkpoint/chatbot")

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry! I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        if tag == "time":
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            for i in list_of_intents:
                if i["tag"] == "time":
                    result = random.choice(i["responses"]) + current_time
                    return result
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result

print("Press 0 if you don't want to chat with my ChatBot.")
while True:
    message = input("")
    if message == "0":
        break
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)