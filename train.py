#!/usr/bin/env python

# train.py
import tensorflow as tf
keras = tf.keras
models = keras.models
layers = keras.layers
optimizers = keras.optimizers
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
'''
from models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
'''
import random
import os

# Ensure necessary NLTK data is downloaded
try:
    print("Checking for NLTK resource 'wordnet'...")
    nltk.data.find('corpora/wordnet')
    print("'wordnet' resource found.")
except LookupError:
    print("NLTK 'wordnet' resource not found. Downloading...")
    nltk.download('wordnet')
try:
    print("Checking for NLTK resource 'punkt'...")
    nltk.data.find('tokenizers/punkt')
    print("'punkt' resource found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')
try:
    print("Checking for NLTK resource 'punkt_tab'...")
    # Attempt to find the resource directory structure
    nltk.data.find('tokenizers/punkt_tab')
    print("'punkt_tab' resource found.")
except LookupError:
    print("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

print("Processing intents...")

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print(f"{len(documents)} documents")
# classes = intents
print(f"{len(classes)} classes: {classes}")
# words = all words, vocabulary
print(f"{len(words)} unique lemmatized words: {words[:20]}...") # Print first 20 words

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create our training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

print("Creating training data...")

# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)

# Separate bag and output_row cleanly
train_x_list = [item[0] for item in training]
train_y_list = [item[1] for item in training]

train_x = np.array(train_x_list)
train_y = np.array(train_y_list)

print("Training data created.")
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
print("Building model...")
model = models.Sequential()
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# Consider trying Adam optimizer as well
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
print("Training model...")
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Ensure the model directory exists if needed (not strictly necessary here)
# if not os.path.exists('model'):
#    os.makedirs('model')

model.save('chatbot_model.h5')

print("Model created and saved as chatbot_model.h5")