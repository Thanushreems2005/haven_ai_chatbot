# chatbot.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datetime import datetime
import random # Added for potential future use or if you add random choices here

# Import functions from your new response manager
from response_manager import get_structured_response, handle_ongoing_exercise # Assuming handle_ongoing_exercise is defined there

# Import the database manager
from db_manager import get_user_preferences, save_user_preferences, init_db # Assuming these functions exist

# --- Global Variables ---
# Load DialoGPT model and tokenizer
print("Loading DialoGPT model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    print("DialoGPT model loaded.")
except Exception as e:
    print(f"Error loading DialoGPT model: {e}. Ensure transformers and torch are installed and model exists.")
    tokenizer = None
    model = None

# --- Initialize Database ---
try:
    init_db()
    print("Database initialized (from chatbot.py).")
except Exception as e:
    print(f"Error initializing database: {e}. Please check db_manager.py.")

# --- Conversation History Store ---
# This will store chat history and exercise state for each session ID
chat_history_ids = {}

# --- Offensive Keywords (for initial filter, can be managed from response_manager too) ---
offensive_keywords = ["fuck", "asshole", "bitch", "cunt", "shit", "damn", "idiot", "stupid", "moron"]

# --- Logging (Basic implementation, could be expanded to a file/DB) ---
def log_interaction(event_type, user_input=None, bot_response=None, ip_address="N/A", intent_tag=None, classifier_score=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Event Type: {event_type}\n"
        f"IP Address (Session ID): {ip_address}\n"
    )
    if user_input is not None:
        log_entry += f"User Input: {user_input}\n"
    if bot_response is not None:
        log_entry += f"Bot Response: {bot_response}\n"
    if intent_tag is not None:
        log_entry += f"Intent Tag: {intent_tag}\n"
    if classifier_score is not None:
        log_entry += f"Classifier Score: {classifier_score:.4f}\n"
    log_entry += "-"*30 + "\n" # Separator for readability

    # For now, print to console. In a real application, write to a log file or database.
    # print(log_entry)
    with open("haven_chat_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

# --- Core Chatbot Logic ---
def get_haven_response(user_input, session_id):
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize session data if it doesn't exist
    if session_id not in chat_history_ids:
        chat_history_ids[session_id] = {
            'history': None,
            'exercise_state': None, # Stores active exercise type and step
            'last_intent': None # Stores the last classified intent for context if needed
        }
        log_interaction("SESSION_INIT", ip_address=session_id)
        print(f"Initialized new session for {session_id}")

    current_session = chat_history_ids[session_id]
    bot_response = ""
    intent_tag = "UNKNOWN" # Default intent tag

    # 1. Check for ongoing exercise first (highest priority for multi-turn)
    if current_session['exercise_state']:
        print(f"Continuing ongoing exercise: {current_session['exercise_state']}")
        response_from_exercise, exercise_complete = handle_ongoing_exercise(user_input, current_session['exercise_state'])

        if response_from_exercise:
            bot_response = response_from_exercise
            # Update history with exercise specific response
            if model and tokenizer: # Only if DialoGPT is loaded
                try:
                    # Append user input and exercise response to history for DialoGPT context
                    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
                    new_bot_response_ids = tokenizer.encode(bot_response + tokenizer.eos_token, return_tensors='pt')

                    if current_session['history'] is None:
                        current_session['history'] = torch.cat([new_user_input_ids, new_bot_response_ids], dim=-1)
                    else:
                        current_session['history'] = torch.cat([current_session['history'], new_user_input_ids, new_bot_response_ids], dim=-1)
                except Exception as e:
                    print(f"Error updating DialoGPT history for exercise: {e}")

            if exercise_complete:
                print(f"Exercise {current_session['exercise_state']['type']} completed.")
                current_session['exercise_state'] = None # Clear exercise state
                intent_tag = f"EXERCISE_COMPLETE_{current_session['exercise_state']['type']}"
                bot_response += "\nIs there anything else I can help you with?" # Add a concluding remark

            log_interaction("EXERCISE_CONTINUATION", user_input=user_input, bot_response=bot_response, ip_address=session_id, intent_tag=intent_tag)
            return bot_response

    # 2. If no ongoing exercise, get structured response
    # get_structured_response now returns (response_text, intent_tag, updated_session_data)
    structured_response, intent_tag, updated_session_data = get_structured_response(user_input)

    if structured_response:
        bot_response = structured_response
        # Apply any session data updates returned by get_structured_response
        current_session.update(updated_session_data)
        # Update DialoGPT history for context, if needed for future turns
        if model and tokenizer:
            try:
                new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
                new_bot_response_ids = tokenizer.encode(bot_response + tokenizer.eos_token, return_tensors='pt')

                if current_session['history'] is None:
                    current_session['history'] = torch.cat([new_user_input_ids, new_bot_response_ids], dim=-1)
                else:
                    current_session['history'] = torch.cat([current_session['history'], new_user_input_ids, new_bot_response_ids], dim=-1)
            except Exception as e:
                print(f"Error updating DialoGPT history for structured response: {e}")
        
        log_interaction("STRUCTURED_RESPONSE", user_input=user_input, bot_response=bot_response, ip_address=session_id, intent_tag=intent_tag)
        return bot_response

    # 3. Fallback to DialoGPT if no structured response is found
    # This block only executes if structured_response is None
    if model is None or tokenizer is None:
        error_message = "I'm sorry, I'm having trouble processing your request right now. My core models are not loaded."
        log_interaction("MODEL_NOT_LOADED", user_input=user_input, bot_response=error_message, ip_address=session_id)
        return error_message

    try:
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        if current_session['history'] is None:
            chat_history_ids[session_id]['history'] = new_user_input_ids
        else:
            chat_history_ids[session_id]['history'] = torch.cat([current_session['history'], new_user_input_ids], dim=-1)

        # Generate a response from the model
        bot_input_ids = chat_history_ids[session_id]['history']
        chat_history_ids[session_id]['history'] = model.generate(
            bot_input_ids,
            max_length=1000, # Increased max_length for longer conversations
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=3 # Helps prevent repetitive phrases
        )
    except Exception as e:
        print(f"Error during DialoGPT generation: {e}")
        log_interaction("DIALGPT_ERROR", user_input=user_input, bot_response=f"Error: {e}", ip_address=session_id)
        return "I'm having a little trouble thinking right now. Could you rephrase that?"

    # Decode the bot's response
    response = tokenizer.decode(
        chat_history_ids[session_id]['history'][:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Basic post-processing for DialoGPT's common quirks
    if not response.strip() or len(response.split()) < 5:
        response = "I'm here to listen. Can you tell me more about what's on your mind?"
    # Filter out specific undesirable phrases often generated by DialoGPT
    elif "i'll have to check your post history" in response.lower() or "i'll have my coffee" in response.lower() or \
         "what are your thoughts on that" in response.lower(): # Added another common undesirable phrase
        response = "I'm here to support you. What else is on your mind?"

    log_interaction("DIALGPT_FALLBACK", user_input=user_input, bot_response=response, ip_address=session_id, intent_tag="DIALGPT")
    return response

def reset_conversation(session_id):
    """Resets the conversation history and exercise state for a given session ID."""
    if session_id in chat_history_ids:
        del chat_history_ids[session_id]
        print(f"Conversation history and exercise state for session {session_id} reset.")
    else:
        print(f"No conversation history found for session {session_id}.")