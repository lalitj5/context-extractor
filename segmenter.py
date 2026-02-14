import json
from config import model, segment_window_size, segment_overlap

def segment_conversation(messages, client):
    if len(messages) < segment_window_size * 2:
        boundaries = detect_boundaries(messages, 0, client)
    else:
        boundaries = windowed_detection(messages, client)

    return assemble_segments(messages, boundaries)


def format_numbered_messages(messages, offset=0):
    lines = []
    for i, msg in enumerate(messages):
        lines.append(f"[{offset + i}] {msg['role']}: {msg['content']}")
    return "\n".join(lines)


def detect_boundaries(messages, offset, client):
    formatted = format_numbered_messages(messages, offset)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this conversation and identify distinct topic segments. Each message is numbered with its index.

Return a JSON array of objects, each with:
- "topic": short description of the topic (5-10 words max)
- "start": index of the first message in this segment
- "end": index of the last message in this segment

Rules:
- Segments must be contiguous — no gaps or overlaps
- The first segment must start at index {offset}
- The last segment must end at index {offset + len(messages) - 1}
- Every message must belong to exactly one segment
- Prefer fewer, meaningful segments over many tiny ones

Conversation:
{formatted}"""
            },
            {
                "role": "assistant",
                "content": "["
            }
        ],
        stop_sequences=["```"]
    )

    raw = "[" + response.content[0].text
    # strip any trailing text after the JSON array
    bracket_depth = 0
    end_pos = 0
    for i, ch in enumerate(raw):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                end_pos = i + 1
                break
    if end_pos > 0:
        raw = raw[:end_pos]

    return json.loads(raw)


def windowed_detection(messages, client):
    all_boundaries = []
    step = segment_window_size - segment_overlap
    pos = 0

    while pos < len(messages):
        window_end = min(pos + segment_window_size, len(messages))
        window = messages[pos:window_end]
        boundaries = detect_boundaries(window, pos, client)
        all_boundaries.append(boundaries)
        if window_end >= len(messages):
            break
        pos += step

    return merge_windows(all_boundaries, len(messages))


def merge_windows(all_windows, total_messages):
    if len(all_windows) == 1:
        return all_windows[0]

    merged = list(all_windows[0])

    for window in all_windows[1:]:
        # find where previous merged segments end
        prev_end = merged[-1]["end"]

        # find first segment in new window that starts after prev_end
        for seg in window:
            if seg["start"] > prev_end:
                merged.append(seg)
            elif seg["end"] > prev_end:
                # partial overlap — extend the last merged segment or start new one
                merged.append({
                    "topic": seg["topic"],
                    "start": prev_end + 1,
                    "end": seg["end"]
                })

    # fix any gaps between segments
    for i in range(1, len(merged)):
        if merged[i]["start"] != merged[i - 1]["end"] + 1:
            merged[i]["start"] = merged[i - 1]["end"] + 1

    # ensure last segment covers to the end
    merged[-1]["end"] = total_messages - 1

    return merged


def assemble_segments(messages, boundaries):
    segments = []
    for i, boundary in enumerate(boundaries):
        start = boundary["start"]
        end = boundary["end"]
        segment_messages = []
        for idx in range(start, end + 1):
            segment_messages.append({
                "index": idx,
                "role": messages[idx]["role"],
                "content": messages[idx]["content"]
            })
        segments.append({
            "segment_id": i + 1,
            "topic": boundary["topic"],
            "start_index": start,
            "end_index": end,
            "message_count": end - start + 1,
            "messages": segment_messages
        })

    validate_segments(segments, len(messages))
    return segments


def validate_segments(segments, total_messages):
    if not segments:
        raise ValueError("No segments produced")

    if segments[0]["start_index"] != 0:
        raise ValueError(f"First segment starts at {segments[0]['start_index']}, expected 0")

    if segments[-1]["end_index"] != total_messages - 1:
        raise ValueError(f"Last segment ends at {segments[-1]['end_index']}, expected {total_messages - 1}")

    for i in range(1, len(segments)):
        expected_start = segments[i - 1]["end_index"] + 1
        actual_start = segments[i]["start_index"]
        if actual_start != expected_start:
            raise ValueError(f"Gap or overlap between segment {i} and {i + 1}: expected start {expected_start}, got {actual_start}")
