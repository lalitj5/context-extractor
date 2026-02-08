# Takes flagged messages and generates a structured summary

# Appends all flagged keywords and corresponding content
def format_messages_for_prompt(flagged_messages):
    lines = []
    for msg in flagged_messages:
        keywords_str = ", ".join(msg["matched_keywords"])
        lines.append(f"[{keywords_str}] {msg['role']}: {msg['content']}")
    return "\n\n".join(lines)


def synthesize(flagged_messages, client):
    if not flagged_messages:
        return "No important context found in the transcript."

    formatted = format_messages_for_prompt(flagged_messages)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"""
                Based on these flagged conversation excerpts, create a structured context summary.
                Organize into these sections (skip any that don't apply):

                ## Key Decisions
                ## Important Facts
                ## Technical Details
                ## Open Questions

                Be concise. This summary will be used to initialize a new conversation.

                Flagged excerpts:
                {formatted}
                """
            }
        ]
    )

    return response.content[0].text
