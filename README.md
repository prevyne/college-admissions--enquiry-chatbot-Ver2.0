Here is a clean, professional README.md file for your project. You can copy this directly into a file named README.md in your project's root directory.

College Admissions Chatbot Assistant
A Python Flask-based chatbot designed to assist prospective students with college admission queries. It uses Natural Language Processing (NLP) and a deep learning model to understand user intents and provide relevant responses, featuring a persistent chat history system.

Features
Intent-Based Responses: Uses a Neural Network (TensorFlow/Keras) trained on custom intents (intents.json) to classify user queries.

Web Interface: Clean, responsive UI built with HTML, CSS, and JavaScript.

Chat Management:

Start new conversations.

Auto-Naming: Chat sessions are automatically titled based on the first user message.

View history of past conversations via a sidebar.

Delete specific chat sessions.

Persistence: Chat history is saved locally as JSON files.

Export: Download chat logs as .txt or .json files.

🛠️ Tech Stack
Backend: Python 3, Flask

ML/NLP: TensorFlow, Keras, NLTK, NumPy

Frontend: HTML5, CSS3, JavaScript (Fetch API)

Data Storage: JSON (File-based)

Project Structure
Plaintext

college_chatbot/
├── app.py              # Main Flask application entry point
├── train.py            # Script to train the Keras model
├── chat_history.py     # Module for handling chat persistence (save/load/delete)
├── intents.json        # Knowledge base (patterns and responses)
├── templates/
│   └── index.html      # Frontend HTML template
├── chat_histories/     # Auto-generated directory for saved chats
├── chatbot_model.h5    # Trained model file (generated)
├── classes.pkl         # Pickled intent classes (generated)
├── words.pkl           # Pickled vocabulary (generated)
└── requirements.txt    # Python dependencies
Installation & Setup
1. Prerequisites
Python 3.8+ installed on your system.

2. Clone/Download the Repository
Navigate to the project folder in your terminal.

3. Create a Virtual Environment (Recommended)
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Bash

pip install -r requirements.txt
Note: If you don't have a requirements.txt yet, create one with the following content:

Plaintext

Flask
nltk
numpy
tensorflow
5. Train the Model
Before running the app, you must train the chatbot to generate the model files (.h5, .pkl).

Bash

python train.py
Wait for the script to finish. It will verify NLTK data downloads and output "Model created and saved as chatbot_model.h5".

6. Run the Application
Bash

python app.py
7. Access the Chatbot
Open your web browser and navigate to: http://127.0.0.1:5000

Customizing the Knowledge Base
To teach the chatbot new things, modify the intents.json file.

Open intents.json.

Add a new JSON object to the intents list:

JSON

{
  "tag": "campus_security",
  "patterns": ["Is the campus safe?", "Security information", "Emergency contacts"],
  "responses": ["Our campus is very safe and monitored 24/7. Campus security can be reached at 555-0199."]
}
Important: You must re-run python train.py whenever you modify intents.json for changes to take effect.

Troubleshooting
"File signature not found" or Model loading errors:

Delete chatbot_model.h5, words.pkl, and classes.pkl.

Re-run python train.py.

NLTK Lookup Errors: The scripts are designed to auto-download missing data (punkt, wordnet, punkt_tab). If this fails, open a Python shell and run:

Python

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

Contributing
Feel free to fork this project and submit pull requests. You can enhance the intents.json file to make the bot smarter or improve the frontend design.

License
This project is open-source and available for educational purposes.
