# chat_history.py

import os
import json
import uuid
import datetime

# Directory where chat history files will be stored
_HISTORY_DIR = "chat_histories"

# --- Helper Functions ---

def _ensure_history_dir():
    """Ensures the chat history directory exists."""
    if not os.path.exists(_HISTORY_DIR):
        try:
            os.makedirs(_HISTORY_DIR)
            print(f"Created chat history directory: {_HISTORY_DIR}")
        except OSError as e:
            print(f"Error creating directory {_HISTORY_DIR}: {e}")
            # Depending on requirements, you might want to raise the exception
            # raise

def _get_history_filepath(chat_id):
    """Constructs the full path for a given chat ID's JSON file."""
    return os.path.join(_HISTORY_DIR, f"{chat_id}.json")

def _get_current_utc_timestamp():
    """Returns the current time in UTC ISO format string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

# --- Public API Functions ---

def create_new_chat():
    """
    Creates a new chat session file.

    Returns:
        str: The unique ID of the newly created chat, or None if creation failed.
    """
    _ensure_history_dir()
    new_id = str(uuid.uuid4())
    filepath = _get_history_filepath(new_id)
    timestamp = _get_current_utc_timestamp()

    initial_data = {
        "chat_id": new_id,
        "created_at": timestamp,
        "title": f"Chat started {timestamp}", # Auto-generated title
        "messages": []
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)
        print(f"Created new chat: {new_id}")
        return new_id
    except IOError as e:
        print(f"Error creating chat file {filepath}: {e}")
        return None

def add_message(chat_id, sender, message):
    """
    Adds a message to a specific chat history file.
    If it's the first user message, it also updates the chat title.

    Args:
        chat_id (str): The ID of the chat to add the message to.
        sender (str): The sender of the message ('user' or 'bot').
        message (str): The content of the message.

    Returns:
        bool: True if the message was added successfully, False otherwise.
    """
    _ensure_history_dir()
    filepath = _get_history_filepath(chat_id)

    if not os.path.exists(filepath):
        print(f"Error: Chat file not found for ID {chat_id}")
        return False

    new_message = {
        "sender": sender,
        "message": message,
        "timestamp": _get_current_utc_timestamp()
    }

    try:
        # Read existing data
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        # --- Automatic Title Generation Logic ---
        # Check if this is the very first message AND it's from the user
        is_first_message_ever = not chat_data.get('messages') # Checks if messages list is empty
        if sender == 'user' and is_first_message_ever:
            # Generate title from the first message content
            max_title_length = 40  # Adjust max length as needed
            if len(message) > max_title_length:
                # Truncate and add ellipsis
                new_title = message[:max_title_length].strip() + "..."
            else:
                new_title = message.strip()

            # Handle cases where the message is empty or whitespace only
            if not new_title:
                new_title = "Untitled Chat" # Provide a fallback

            # Update the title in the chat data
            chat_data['title'] = new_title
            print(f"Automatically set title for chat {chat_id} to: '{new_title}'")
        # --- End Automatic Title Generation Logic ---

        # Append new message (ensure 'messages' key exists)
        chat_data.setdefault('messages', []).append(new_message)

        # Write updated data back
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2)
        return True

    except (IOError, json.JSONDecodeError, KeyError) as e:
        print(f"Error updating chat file {filepath}: {e}")
        return False

def get_chat_history(chat_id):
    """
    Retrieves the entire chat history for a given chat ID.

    Args:
        chat_id (str): The ID of the chat to retrieve.

    Returns:
        dict: The chat history data (including metadata and messages),
              or None if the chat is not found or an error occurs.
    """
    _ensure_history_dir()
    filepath = _get_history_filepath(chat_id)

    if not os.path.exists(filepath):
        # print(f"Chat history not found for ID: {chat_id}") # Optional: less verbose
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        return chat_data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading chat file {filepath}: {e}")
        return None

def list_chats():
    """
    Lists available chat histories based on the files in the history directory.

    Returns:
        list: A list of dictionaries, each containing {'chat_id', 'title', 'created_at'}
              for a saved chat, sorted by creation date (newest first).
              Returns an empty list if the directory doesn't exist or is empty.
    """
    _ensure_history_dir()
    chat_list = []

    if not os.path.exists(_HISTORY_DIR):
        return []

    try:
        filenames = os.listdir(_HISTORY_DIR)
    except OSError as e:
        print(f"Error listing directory {_HISTORY_DIR}: {e}")
        return []

    for filename in filenames:
        if filename.endswith(".json"):
            chat_id = filename[:-5] # Remove '.json' extension
            filepath = _get_history_filepath(chat_id)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Only read necessary metadata to avoid loading large message lists
                    data = json.load(f)
                    chat_info = {
                        "chat_id": data.get("chat_id", chat_id), # Use data's ID if available
                        "title": data.get("title", f"Chat {chat_id}"), # Default title
                        "created_at": data.get("created_at", "N/A")
                    }
                    chat_list.append(chat_info)
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Skipping corrupted or invalid chat file {filepath}: {e}")
                # Optionally add placeholder? {"chat_id": chat_id, "title": "Error Loading", "created_at": "N/A"}

    # Sort by creation date, newest first (assuming ISO format sorts correctly)
    chat_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    return chat_list

def delete_chat(chat_id):
    """
    Deletes a chat history file.

    Args:
        chat_id (str): The ID of the chat to delete.

    Returns:
        bool: True if deletion was successful or file didn't exist, False on error.
    """
    _ensure_history_dir()
    filepath = _get_history_filepath(chat_id)

    if not os.path.exists(filepath):
        print(f"Chat file for ID {chat_id} not found. Nothing to delete.")
        return True # Or False depending on desired behavior for non-existent files

    try:
        os.remove(filepath)
        print(f"Deleted chat file: {filepath}")
        return True
    except OSError as e:
        print(f"Error deleting chat file {filepath}: {e}")
        return False

def export_chat(chat_id, format='txt'):
    """
    Exports a chat history in the specified format.

    Args:
        chat_id (str): The ID of the chat to export.
        format (str): The desired format ('txt' or 'json'). Defaults to 'txt'.

    Returns:
        str: The formatted chat history as a string, or None if the chat
             is not found or an error occurs.
    """
    chat_data = get_chat_history(chat_id)
    if not chat_data:
        return None

    if format.lower() == 'json':
        try:
            # Pretty-print the JSON data
            return json.dumps(chat_data, indent=2)
        except TypeError as e:
            print(f"Error serializing chat data to JSON for {chat_id}: {e}")
            return None # Should not happen with standard structure

    elif format.lower() == 'txt':
        try:
            export_lines = []
            export_lines.append(f"Chat History: {chat_data.get('title', chat_id)}")
            export_lines.append(f"Chat ID: {chat_data.get('chat_id', 'N/A')}")
            export_lines.append(f"Created At: {chat_data.get('created_at', 'N/A')}")
            export_lines.append("\n" + "="*20 + "\n") # Separator

            messages = chat_data.get('messages', [])
            if not messages:
                export_lines.append("No messages in this chat.")
            else:
                for msg in messages:
                    sender = msg.get('sender', 'Unknown').capitalize()
                    timestamp = msg.get('timestamp', 'No Timestamp')
                    message_text = msg.get('message', '')
                    export_lines.append(f"[{timestamp}] {sender}: {message_text}")

            return "\n".join(export_lines)
        except Exception as e: # Catch broader exceptions during formatting
            print(f"Error formatting chat data to TXT for {chat_id}: {e}")
            return None
    else:
        print(f"Unsupported export format: {format}")
        return None # Or default to TXT export

# --- Example Usage (Optional: For testing the module directly) ---
if __name__ == "__main__":
    print("Testing chat_history module...")

    # Test create
    new_id1 = create_new_chat()
    new_id2 = create_new_chat()
    print(f"Created chat IDs: {new_id1}, {new_id2}")

    # Test add message
    if new_id1:
        add_message(new_id1, "user", "Hello bot!")
        add_message(new_id1, "bot", "Hello user!")
        add_message(new_id1, "user", "How are you?")

    # Test list
    chats = list_chats()
    print("\nAvailable chats:")
    for chat in chats:
        print(f"- ID: {chat['chat_id']}, Title: {chat['title']}, Created: {chat['created_at']}")

    # Test get history
    if new_id1:
        history = get_chat_history(new_id1)
        if history:
            print(f"\nHistory for {new_id1}:")
            print(json.dumps(history, indent=2))
        else:
            print(f"\nCould not retrieve history for {new_id1}")

    # Test export
    if new_id1:
        txt_export = export_chat(new_id1, format='txt')
        if txt_export:
            print(f"\nTXT Export for {new_id1}:\n---\n{txt_export}\n---")
        json_export = export_chat(new_id1, format='json')

    print("\nTesting complete.")