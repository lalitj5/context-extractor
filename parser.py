import json


def load_transcript(filepath):
    # opens the file and ensures no funny characters    
    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)

    if not isinstance(messages, list):
        raise ValueError("Transcript must be a JSON array of messages")

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' fields")

    return messages
