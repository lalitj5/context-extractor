from config import chunk_size

# for every ten messages, a chunk is made
def chunk_messages(messages):
    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunks.append(messages[i:i + chunk_size])
    return chunks

# All of the messages are appended into a paragraph
# The messages will be in JSON, so we also distinguish
# between the user and the model
def format_chunk_for_prompt(chunk):
    lines = []
    for msg in chunk:
        lines.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(lines)

# Combines functions from above
def extract_keywords(messages, client):
    chunks = chunk_messages(messages)
    all_keywords = []

    for chunk in chunks:
        conversation_text = format_chunk_for_prompt(chunk)

        # calling the anthropic API
        # prefill forces Claude to start with our format
        # stop_sequences ends generation after the keywords
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            stop_sequences=["---"],
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Extract 3-5 important keywords or short phrases from this conversation chunk.
                    Return only the keywords, one per line, no numbering or bullets.
                    End with --- when done.

                    Conversation:
                    {conversation_text}
                    """
                },
                # this is the message prefill, which tricks the model
                # into generating just our keywords
                {
                    "role": "assistant",
                    "content": "Keywords:"
                }
            ]
        )

        keywords_text = response.content[0].text
        keywords = [k.strip() for k in keywords_text.strip().split("\n") if k.strip()]
        all_keywords.extend(keywords)

    return all_keywords
