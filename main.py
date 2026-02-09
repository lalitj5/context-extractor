import argparse
import anthropic

from config import ANTHROPIC_API_KEY
from parser import load_transcript
from keyword_extractor import extract_keywords
from message_flagger import flag_messages
from synthesizer import synthesize


def main():
    # parse command line arguments
    arg_parser = argparse.ArgumentParser(
        description="Extract context from chat transcripts"
    )
    arg_parser.add_argument("transcript", help="Path to JSON transcript file")
    arg_parser.add_argument("-o", "--output", help="Output file (default: print to console)")
    args = arg_parser.parse_args()

    # check for API key
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return

    # initialize anthropic api
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # load transcript
    messages = load_transcript(args.transcript)
    print(f"Loaded {len(messages)} messages")

    # extract keywords
    keywords = extract_keywords(messages, client)
    print(f"Found {len(keywords)} keywords: {keywords}")

    # flag messages
    flagged = flag_messages(messages, keywords)
    print(f"Flagged {len(flagged)} messages")

    # synthesize summary
    summary = synthesize(flagged, client)

    # output result
    # (output file provided) ? writes in file : writes summary in terminal
    # ternary if-else documentation pog
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary written to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
