# haven_ai_chatbot
Haven AI: Your Mental Wellness Companion
Haven AI is a Flask-based chatbot designed to provide empathetic and structured support for mental well-being. It leverages a zero-shot classification model to understand user intent and deliver relevant, supportive responses, with a fallback to a generative model (DialoGPT) for more conversational interactions.

Features
Intent-Based Responses: Utilizes a zero-shot classification model (facebook/bart-large-mnli) to categorize user input into specific mental health-related intents.
Crisis Intervention: Prioritizes and provides immediate crisis resources for high-alert intents (e.g., suicidal thoughts, self-harm).
Multi-Turn Exercises: Guides users through structured coping mechanisms and exercises (e.g., STOP method for anger, 4-7-8 breathing for panic, 5-4-3-2-1 grounding for anxiety).
Conversational Fallback: If a specific intent isn't confidently detected, the chatbot gracefully falls back to a generative model (microsoft/DialoGPT-medium) for more general conversation.
User-Friendly Interface: Features a modern, dark-mode web interface built with HTML and CSS for an engaging user experience.
Session Management: Maintains conversation history and exercise states for individual users.
Basic Logging: Records interactions for review and analysis.
SQLite Database: Stores user preferences (though currently minimally utilized, it provides a foundation for personalization).

Setup
Follow these steps to get Haven AI up and running on your local machine.
Prerequisites
Python 3.8+
pip (Python package installer)

Installation
Download the files: Ensure you have app.py, chatbot.py, response_manager.py, and db_manager.py in the same directory.
Install dependencies: Open your terminal or command prompt and navigate to the directory where you saved the files. Then, run:

pip install Flask transformers torch
Note: torch might require specific installation steps depending on your system and GPU availability. Refer to the PyTorch website for detailed instructions if you encounter issues.

Running the Application
Start the Flask server: In your terminal, from the project directory, run:

python app.py
Access the chatbot: Open your web browser and go to the address printed in your terminal (e.g., http://0.0.0.0:7860/ or http://127.0.0.1:7860/).

Usage
Start a conversation: The chatbot will greet you upon loading.
Express your feelings: Try phrases like "I'm feeling stressed," "I'm sad," "I have anxiety," or "I'm feeling burned out."
Request help: Ask for specific support like "Tell me a joke," "Give me an exercise," or "How can I manage my anger?"
Engage in exercises: If an exercise is initiated, follow the prompts. You can usually type "stop" or "done" to end an exercise.
Reset the chat: Click the "Reset Chat" button to clear the conversation history and start fresh.

File Structure
app.py: The main Flask application file. Handles web routes, serves the HTML interface, and integrates with the chatbot logic.
chatbot.py: Contains the core chatbot logic, including session management, integration with the response manager, and the DialoGPT generative model.
response_manager.py: Manages intent classification, defines specific intent handlers, crisis detection, and multi-turn exercise logic. This is where the "intelligence" for structured responses resides.
db_manager.py: Handles basic SQLite database operations for user preferences (e.g., haven_users.db).

Important Notes
Model Loading: The transformers models (bart-large-mnli and DialoGPT-medium) are downloaded the first time the application runs. This might take some time and require an internet connection.
Crisis Threshold: The crisis_threshold in response_manager.py determines how sensitive the chatbot is to crisis-related phrases. It's currently set to 0.75. You can adjust this value if you find the chatbot is too sensitive or not sensitive enough to crisis inputs.
No alert() or confirm(): The web interface avoids using standard browser alert() and confirm() dialogs as they do not function correctly within the Canvas environment.
