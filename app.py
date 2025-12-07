#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os

import chat_history 

app = Flask(__name__)
# Set a secret key for session management
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

print("Loading AI Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
print("Model loaded successfully.")

# ---THE KNOWLEDGE BASE---

data = {
    'text': [
        # --- Graduation ---
        "When is graduation?", "When is the ceremony?", "When do I graduate?", 
        "graduation date", "commencement schedule",
        # --- Exams ---
        "When are exams?", "dates for finals", "test schedule", 
        "when are tests", "exam calendar", "final exam dates",
        # --- Greetings ---
        "Hi", "Hello", "Good morning", "Hey there", "Is anyone there?",
        # --- Programs ---
        "What courses do you offer?", "list programs", "available majors", 
        "Do you have computer science?", "engineering degrees",
        # --- Tuition ---
        "How much is tuition?", "cost of fees", "price to study here", 
        "tuition fees", "financial cost",
        # --- Application ---
        "When is the application deadline?", "due date for applying", 
        "application requirements", "how to apply"
    ],
    'intent': [
        "graduation", "graduation", "graduation", "graduation", "graduation",
        "exams", "exams", "exams", "exams", "exams", "exams",
        "greeting", "greeting", "greeting", "greeting", "greeting",
        "programs", "programs", "programs", "programs", "programs",
        "tuition", "tuition", "tuition", "tuition", "tuition",
        "application", "application", "application", "application"
    ],
    'response': [
        "Graduation is held in mid-May for Spring and mid-December for Fall.", 
        "Graduation is held in mid-May for Spring and mid-December for Fall.", 
        "Graduation is held in mid-May for Spring and mid-December for Fall.", 
        "Graduation is held in mid-May for Spring and mid-December for Fall.", 
        "Graduation is held in mid-May for Spring and mid-December for Fall.",

        "Final exams start on December 10th and end on December 18th.",
        "Final exams start on December 10th and end on December 18th.",
        "Final exams start on December 10th and end on December 18th.",
        "Final exams start on December 10th and end on December 18th.",
        "Final exams start on December 10th and end on December 18th.",
        "Final exams start on December 10th and end on December 18th.",

        "Hello! I can help you with Graduation dates, Exams, Tuition, and Programs.",
        "Hello! I can help you with Graduation dates, Exams, Tuition, and Programs.",
        "Hello! I can help you with Graduation dates, Exams, Tuition, and Programs.",
        "Hello! I can help you with Graduation dates, Exams, Tuition, and Programs.",
        "Hello! I can help you with Graduation dates, Exams, Tuition, and Programs.",

        "We offer Computer Science, Engineering, Business, and Arts.",
        "We offer Computer Science, Engineering, Business, and Arts.",
        "We offer Computer Science, Engineering, Business, and Arts.",
        "We offer Computer Science, Engineering, Business, and Arts.",
        "We offer Computer Science, Engineering, Business, and Arts.",

        "Tuition is approximately $5,000 per semester for in-state students.",
        "Tuition is approximately $5,000 per semester for in-state students.",
        "Tuition is approximately $5,000 per semester for in-state students.",
        "Tuition is approximately $5,000 per semester for in-state students.",
        "Tuition is approximately $5,000 per semester for in-state students.",
        
        "The Fall application deadline is January 15th.",
        "The Fall application deadline is January 15th.",
        "The Fall application deadline is January 15th.",
        "The Fall application deadline is January 15th."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# --- ENCODE KNOWLEDGE BASE ---
print("Encoding Knowledge Base (Creating Vectors)...")
# Convert all 'text' examples into vector embeddings
knowledge_vectors = model.encode(df['text'].tolist())
print("Knowledge Base Ready!")

# --- SMART RESPONSE LOGIC ---
def get_smart_response(user_input):
    """
    Finds the most similar question in our database and returns the corresponding response.
    """
    user_vector = model.encode([user_input])

    similarities = cosine_similarity(user_vector, knowledge_vectors)[0]
    
    # Find the best match
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    if best_score < 0.40:
        return "I'm not sure about that. Could you rephrase? I can discuss Exams, Graduation, or Tuition."
    
    # Return the response associated with the matched question
    return df.iloc[best_index]['response']

# --- FLASK ROUTES ---

@app.route("/")
def home():
    """Renders the main chat page."""
    current_chat_id = session.get('current_chat_id')
    messages = []
    chat_title = "New Chat"

    if current_chat_id:
        chat_data = chat_history.get_chat_history(current_chat_id)
        if chat_data:
            messages = chat_data.get('messages', [])
            chat_title = chat_data.get('title', f"Chat {current_chat_id}")
        else:
            session.pop('current_chat_id', None)
            current_chat_id = None

    chat_list = chat_history.list_chats()
    return render_template("index.html",
                           messages=messages,
                           chat_list=chat_list,
                           current_chat_id=current_chat_id,
                           chat_title=chat_title)

@app.route("/get")
def get_bot_response():
    """Handles user input and returns smart bot response."""
    userText = request.args.get('msg')
    chat_id = session.get('current_chat_id')

    if not userText:
        return jsonify({"response": "Error: Empty message."})

    # ---Ensure Chat ID exists---
    if not chat_id:
        chat_id = chat_history.create_new_chat()
        session['current_chat_id'] = chat_id

    # ---Save User Message---
    chat_history.add_message(chat_id, 'user', userText)

    # ---Get AI Response (The new logic)---
    bot_response = get_smart_response(userText)

    # ---Save Bot Response ---
    chat_history.add_message(chat_id, 'bot', bot_response)

    return jsonify({"response": bot_response, "chat_id": chat_id})

# --- HISTORY ROUTES (Kept from your original code) ---

@app.route("/new_chat", methods=['POST'])
def new_chat():
    new_chat_id = chat_history.create_new_chat()
    if new_chat_id:
        session['current_chat_id'] = new_chat_id
    return redirect(url_for('home'))

@app.route("/load_chat/<string:chat_id>", methods=['GET'])
def load_chat(chat_id):
    if chat_history.get_chat_history(chat_id):
         session['current_chat_id'] = chat_id
    return redirect(url_for('home'))

@app.route("/delete_chat/<string:chat_id>", methods=['POST'])
def delete_chat_route(chat_id):
    chat_history.delete_chat(chat_id)
    if session.get('current_chat_id') == chat_id:
        session.pop('current_chat_id', None)
    return redirect(url_for('home'))

@app.route("/export_chat/<string:chat_id>", methods=['GET'])
def export_chat_route(chat_id):
    format = request.args.get('format', 'txt').lower()
    exported_data = chat_history.export_chat(chat_id, format=format)
    if exported_data is None:
        return "Error: Chat not found.", 404
    
    filename = f"chat_{chat_id}.{format}"
    mimetype = 'text/plain' if format == 'txt' else 'application/json'
    response = make_response(exported_data)
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Content-Type'] = mimetype
    return response

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting Flask server on Colab...")
    # --- Using run_simple or standard run depending on environment ---
    app.run(host='0.0.0.0', port=5000, debug=True)