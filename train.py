import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


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

model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]), ), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
model.summary()
model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)

model.save("./checkpoint/chatbot")