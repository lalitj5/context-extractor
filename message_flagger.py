# Scans messages for keyword matches and returns only flagged ones
# This will be used for only returning flagged messages
# when creating context profiles.

def flag_messages(messages, keywords):
    flagged_messages = []

    for msg in messages:
        content_lower = msg["content"].lower()
        matched = []

        # check if any keyword appears in the message
        for keyword in keywords:
            if keyword.lower() in content_lower:
                matched.append(keyword)

        # only keep messages that matched keywords
        if matched:
            flagged_msg = msg.copy()
            flagged_msg["matched_keywords"] = matched
            flagged_messages.append(flagged_msg)

    return flagged_messages
