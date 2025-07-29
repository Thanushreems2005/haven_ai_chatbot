# db_manager.py

import sqlite3
import json
import os

DATABASE_NAME = 'haven_users.db'

def init_db():
    """Initializes the SQLite database for user preferences."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id TEXT PRIMARY KEY,
                preferences TEXT -- Stores preferences as a JSON string
            )
        ''')
        conn.commit()
        print(f"Database '{DATABASE_NAME}' initialized.")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def get_user_preferences(session_id):
    """Retrieves user preferences for a given session_id."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT preferences FROM user_preferences WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None # No preferences found for this session_id
    except sqlite3.Error as e:
        print(f"Error retrieving preferences for {session_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_user_preferences(session_id, **kwargs):
    """
    Saves or updates user preferences for a given session_id.
    Preferences are stored as a JSON string.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Get existing preferences or start with default
        current_prefs = get_user_preferences(session_id)
        if current_prefs is None:
            current_prefs = {"preferred_exercise_type": None, "last_mood_reported": None}

        # Update preferences with new kwargs
        current_prefs.update(kwargs)

        preferences_json = json.dumps(current_prefs)

        cursor.execute(
            "INSERT OR REPLACE INTO user_preferences (session_id, preferences) VALUES (?, ?)",
            (session_id, preferences_json)
        )
        conn.commit()
        # print(f"Preferences for {session_id} saved: {current_prefs}") # For debugging
    except sqlite3.Error as e:
        print(f"Error saving preferences for {session_id}: {e}")
    finally:
        if conn:
            conn.close()

# Example usage (for testing db_manager.py directly)
if __name__ == '__main__':
    # Ensure a clean slate for testing
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"Removed existing {DATABASE_NAME}")

    init_db()

    # Test saving preferences
    save_user_preferences("test_session_1", preferred_exercise_type="breathing", last_mood_reported="anxious")
    save_user_preferences("test_session_2", last_mood_reported="happy")

    # Test retrieving preferences
    prefs1 = get_user_preferences("test_session_1")
    print(f"Test Session 1 Prefs: {prefs1}")

    prefs2 = get_user_preferences("test_session_2")
    print(f"Test Session 2 Prefs: {prefs2}")

    prefs_non_existent = get_user_preferences("non_existent_session")
    print(f"Non-existent Session Prefs: {prefs_non_existent}")

    # Test updating preferences
    save_user_preferences("test_session_1", last_mood_reported="calmer")
    prefs1_updated = get_user_preferences("test_session_1")
    print(f"Test Session 1 Updated Prefs: {prefs1_updated}")

    # Test creating new on update
    save_user_preferences("test_session_3", preferred_exercise_type="mindfulness")
    prefs3 = get_user_preferences("test_session_3")
    print(f"Test Session 3 Prefs: {prefs3}")