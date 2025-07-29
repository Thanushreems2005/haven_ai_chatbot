# main.py

import response_manager
import datetime

# This will store the state of any ongoing multi-turn exercises for the current session.
# For a real multi-user bot, this would need to be stored per user in a database or similar.
current_exercise_state = {}

def run_chatbot():
    print("Welcome to your Mental Health Support Bot! (Type 'exit' to quit)")
    print("-" * 40)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye! Take care.")
            break

        # Convert user input to lower for initial checks, but pass original to response_manager
        user_input_lower = user_input.lower()

        response_text = ""
        intent_tag = ""
        # Check if an exercise is ongoing
        if current_exercise_state and current_exercise_state.get('active'):
            # Pass user input to the exercise handler
            response_text, is_complete = response_manager.handle_ongoing_exercise(user_input, current_exercise_state)
            if is_complete:
                current_exercise_state['active'] = False # Mark exercise as complete
            intent_tag = current_exercise_state.get('type') # Use exercise type as intent
        else:
            # If no exercise, use the main classification logic
            response_text, intent_tag = response_manager.get_structured_response(user_input)

            # Check if the returned intent initiates an exercise
            if intent_tag == "EXERCISE_PANIC_STEP1":
                current_exercise_state.update({'active': True, 'type': 'COPING_PANIC', 'step': 1})
                # Re-call the handler to get the first step's prompt, as get_structured_response might return the *start* of the exercise.
                response_text, _ = response_manager.handle_ongoing_exercise(user_input, current_exercise_state) # User input is still relevant for the first step
            elif intent_tag == "EXERCISE_ANGER_STEP1":
                current_exercise_state.update({'active': True, 'type': 'COPING_ANGER', 'step': 1})
                response_text, _ = response_manager.handle_ongoing_exercise(user_input, current_exercise_state)
            elif intent_tag == "EXERCISE_ADHD_STEP1":
                current_exercise_state.update({'active': True, 'type': 'COPING_ADHD', 'step': 1})
                response_text, _ = response_manager.handle_ongoing_exercise(user_input, current_exercise_state)


        print(f"Bot: {response_text}")

if __name__ == "__main__":
    run_chatbot()