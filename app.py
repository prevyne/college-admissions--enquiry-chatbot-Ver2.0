# app.py

#from tensorflow.keras.models import load_model
import tensorflow as tf
keras=tf.keras

models=keras.models
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import os # Needed for secret key generation

# --- Flask Imports ---
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response

# --- Chat History Import ---
import chat_history # Import the new module

# --- NLTK Setup (Ensure resources are available) ---
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
    nltk.data.find('tokenizers/punkt_tab') # Check added previously
    print("'punkt_tab' resource found.")
except LookupError:
    print("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab') # Download added previously

# --- Load Chatbot Model and Data ---
lemmatizer = WordNetLemmatizer()
try:
    model = models.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    print("Chatbot resources loaded successfully.")
except Exception as e:
    print(f"FATAL Error loading chatbot resources: {e}")
    print("Please make sure you have run train.py successfully first.")
    exit() # Exit if core resources aren't loaded

# --- Flask App Initialization ---
app = Flask(__name__)

# IMPORTANT: Set a secret key for session management!
# Use an environment variable in production for security.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
print(f"Flask secret key set {'from environment' if 'FLASK_SECRET_KEY' in os.environ else 'randomly'}.")


# --- NLP Processing Functions ---

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False): # Disabled show_details for production
    """Creates a bag of words vector for the sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}") # Keep for debugging if needed
    return np.array(bag)

def predict_class(sentence, model):
    """Predicts the intent class for the sentence."""
    p = bow(sentence, words, show_details=False)
    p_batch = np.expand_dims(p, axis=0)
    res = model.predict(p_batch, verbose=0)[0] # Set verbose=0 to hide prediction progress bar

    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    if not return_list:
         return_list.append({"intent": "fallback", "probability": "1.0"})

    return return_list

def getResponse(ints, intents_json):
    """Gets a random response from the predicted intent."""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    response = "Sorry, I encountered an issue finding a response."
    for i in list_of_intents:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            break
    return response

# --- Flask Routes ---

@app.route("/")
def home():
    """Renders the main chat page, loading current chat if available."""
    current_chat_id = session.get('current_chat_id')
    messages = []
    chat_title = "New Chat" # Default title

    if current_chat_id:
        print(f"Session has current_chat_id: {current_chat_id}")
        chat_data = chat_history.get_chat_history(current_chat_id)
        if chat_data:
            messages = chat_data.get('messages', [])
            chat_title = chat_data.get('title', f"Chat {current_chat_id}") # Use stored title
        else:
            # Chat ID in session is invalid (e.g., file deleted), clear it
            print(f"Chat data not found for ID {current_chat_id}. Clearing from session.")
            session.pop('current_chat_id', None)
            current_chat_id = None # Reset for template context
            chat_title = "New Chat" # Reset title

    # Always get the list of chats to display
    chat_list = chat_history.list_chats()
    print(f"Rendering home page. Current chat: {current_chat_id}. Chat list count: {len(chat_list)}")

    return render_template("index.html",
                           messages=messages,
                           chat_list=chat_list,
                           current_chat_id=current_chat_id,
                           chat_title=chat_title)

@app.route("/get")
def get_bot_response():
    """Handles incoming user messages, saves conversation, returns bot response."""
    userText = request.args.get('msg')
    chat_id = session.get('current_chat_id')

    if not userText:
        return jsonify({"response": "Error: No message received.", "chat_id": chat_id})

    # If no active chat in session, create one first
    if not chat_id:
        print("No chat_id in session, creating a new one...")
        chat_id = chat_history.create_new_chat()
        if not chat_id:
            print("Error: Failed to create new chat session from /get endpoint.")
            return jsonify({"response": "Error: Could not start a new chat session.", "chat_id": None})
        session['current_chat_id'] = chat_id
        print(f"Created and set new chat session: {chat_id}")

    print(f"[Chat {chat_id}] User: {userText}")

    # Save user message
    if not chat_history.add_message(chat_id, 'user', userText):
        print(f"[Chat {chat_id}] Warning: Failed to save user message.")
        # Decide if we should stop? For now, we continue.

    # Get bot response
    ints = predict_class(userText, model)
    bot_response = getResponse(ints, intents)
    print(f"[Chat {chat_id}] Bot: {bot_response}")

    # Save bot response
    if not chat_history.add_message(chat_id, 'bot', bot_response):
         print(f"[Chat {chat_id}] Warning: Failed to save bot response.")
         # Decide if we should stop? For now, we continue.

    # Return response and chat_id (especially useful if a new chat was created)
    return jsonify({"response": bot_response, "chat_id": chat_id})

# --- Chat Management Routes ---

@app.route("/new_chat", methods=['POST'])
def new_chat():
    """Creates a new chat session and redirects to home to load it."""
    print("Received request for /new_chat")
    new_chat_id = chat_history.create_new_chat()
    if new_chat_id:
        session['current_chat_id'] = new_chat_id
        print(f"Set current chat via /new_chat: {new_chat_id}")
    else:
        print("Error: Failed to create new chat via /new_chat endpoint.")
        # Optional: Add Flask flash messaging here to inform the user
    return redirect(url_for('home')) # Redirect back to the main page

@app.route("/load_chat/<string:chat_id>", methods=['GET'])
def load_chat(chat_id):
    """Sets the current chat ID in the session and redirects home."""
    print(f"Received request to load chat: {chat_id}")
    # Check if chat exists before setting it in session
    if chat_history.get_chat_history(chat_id):
         session['current_chat_id'] = chat_id
         print(f"Set current chat via /load_chat: {chat_id}")
    else:
         print(f"Warning: Attempted to load non-existent chat ID: {chat_id}")
         session.pop('current_chat_id', None) # Ensure no invalid ID stays
         # Optional: flash a message 'Chat not found'
    return redirect(url_for('home'))

@app.route("/delete_chat/<string:chat_id>", methods=['POST'])
def delete_chat_route(chat_id):
    """Deletes a specified chat history."""
    print(f"Received request to delete chat: {chat_id}")
    deleted = chat_history.delete_chat(chat_id)

    if deleted:
        print(f"Successfully deleted chat: {chat_id}")
        # If the deleted chat was the active one, clear it from session
        if session.get('current_chat_id') == chat_id:
            session.pop('current_chat_id', None)
            print("Cleared deleted chat ID from session.")
    else:
         print(f"Failed to delete chat: {chat_id}")
         # Optional: flash message 'Failed to delete chat'

    # Redirect back home to refresh the view
    return redirect(url_for('home'))

@app.route("/export_chat/<string:chat_id>", methods=['GET'])
def export_chat_route(chat_id):
    """Exports a chat history as a downloadable file (TXT or JSON)."""
    print(f"Received request to export chat: {chat_id}")
    format = request.args.get('format', 'txt').lower() # Default to txt
    exported_data = chat_history.export_chat(chat_id, format=format)

    if exported_data is None:
        print(f"Export failed: Chat {chat_id} not found or export error.")
        return "Error: Chat not found or export failed.", 404

    filename = f"chat_{chat_id}.{format}"
    mimetype = 'text/plain' if format == 'txt' else 'application/json'

    response = make_response(exported_data)
    # Set headers for download
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Content-Type'] = mimetype
    print(f"Exporting chat {chat_id} as {filename} ({mimetype})")
    return response

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Flask development server...")
    # Use host='0.0.0.0' to make accessible on your network
    # Use debug=True for development (auto-reloads, detailed errors), False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
