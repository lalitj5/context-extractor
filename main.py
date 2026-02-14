import argparse
import json
import os
import anthropic

from config import ANTHROPIC_API_KEY
from parser import load_transcript

# this was the old main.py
def run_legacy(messages, client, output_path):
    from legacy_methods.keyword_extractor import extract_keywords
    from legacy_methods.message_flagger import flag_messages
    from legacy_methods.synthesizer import synthesize

    keywords = extract_keywords(messages, client)
    print(f"Found {len(keywords)} keywords: {keywords}")

    flagged = flag_messages(messages, keywords)
    print(f"Flagged {len(flagged)} messages")

    summary = synthesize(flagged, client)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary written to {output_path}")
    else:
        print(summary)


def run_segment(messages, client, output_path, transcript_path, select_arg):
    from segmenter import segment_conversation

    segments = segment_conversation(messages, client)
    print(f"Found {len(segments)} segments:\n")

    for seg in segments:
        print(f"  [{seg['segment_id']}] {seg['topic']} ({seg['message_count']} messages, {seg['start_index']}-{seg['end_index']})")

    # determine which segments to select
    if select_arg == "all":
        selected_ids = [seg["segment_id"] for seg in segments]
    elif select_arg:
        selected_ids = [int(x.strip()) for x in select_arg.split(",")]
    else:
        print(f"\nSelect segments (e.g. 1,3 or all): ", end="")
        choice = input().strip()
        if choice.lower() == "all":
            selected_ids = [seg["segment_id"] for seg in segments]
        else:
            selected_ids = [int(x.strip()) for x in choice.split(",")]

    selected_segments = [seg for seg in segments if seg["segment_id"] in selected_ids]

    if not selected_segments:
        print("No valid segments selected.")
        return

    output = {
        "source_file": os.path.basename(transcript_path),
        "total_messages": len(messages),
        "selected_segments": len(selected_segments),
        "total_segments": len(segments),
        "segments": selected_segments
    }

    output_json = json.dumps(output, indent=2, ensure_ascii=False)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nOutput written to {output_path}")
    else:
        print(f"\n{output_json}")


def main():
    arg_parser = argparse.ArgumentParser(
        description="Extract context from chat transcripts"
    )
    arg_parser.add_argument("transcript", help="Path to JSON transcript file")
    arg_parser.add_argument("-o", "--output", help="Output file (default: print to console)")
    arg_parser.add_argument("--mode", choices=["segment", "legacy"], default="segment",
                            help="Pipeline mode (default: segment)")
    arg_parser.add_argument("--select", help="Segment selection: comma-separated IDs or 'all' (skips interactive prompt)")
    args = arg_parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = load_transcript(args.transcript)
    print(f"Loaded {len(messages)} messages")

    if args.mode == "legacy":
        run_legacy(messages, client, args.output)
    else:
        run_segment(messages, client, args.output, args.transcript, args.select)


if __name__ == "__main__":
    main()
