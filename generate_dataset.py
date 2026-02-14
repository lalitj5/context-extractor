import argparse
import json
import anthropic

from config import ANTHROPIC_API_KEY


SCENARIOS = [
    "A developer and an AI debugging a tricky race condition in a multithreaded Python application, where they initially misdiagnose the problem before finding the real cause.",
    "A founder discussing their startup pivot with an AI — they started with a B2B SaaS idea but market feedback is pushing them toward B2C, and they're conflicted about the tradeoffs.",
    "A student learning about database design who keeps making wrong assumptions that the AI has to gently correct, covering normalization, indexing, and query optimization.",
    "A team lead planning a system migration from a monolith to microservices, weighing risks, rollback strategies, and which services to extract first.",
    "A designer and AI iterating on a mobile app's UX flow, going back and forth on navigation patterns, with the user changing their mind multiple times based on new constraints.",
    "A student learning calculus and using the chat to complete their homework. The student also works on parts of his economics homework, which is unrelated to the other topic."
]


def generate_conversation(client, scenario, num_turns):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": f"""Generate a realistic chat transcript as a JSON array.

Scenario: {scenario}

Requirements:
- Exactly {num_turns} messages, alternating "user" and "assistant" roles (starting with "user")
- Each message must have "role" and "content" fields
- Make the conversation feel natural — include topic shifts, corrections, callbacks to earlier points, and evolving decisions
- The user should sometimes change their mind, ask follow-ups, or push back
- Include specific technical details, names, or numbers where relevant
- Keep individual messages concise (1-3 sentences typically)

Return ONLY the JSON array, no other text."""
            }
        ]
    )

    text = response.content[0].text.strip()
    # handle markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Generate synthetic chat transcripts for testing"
    )
    arg_parser.add_argument(
        "-n", "--num-chats", type=int, default=3,
        help="Number of conversations to generate (default: 3)"
    )
    arg_parser.add_argument(
        "-t", "--turns", type=int, default=20,
        help="Number of messages per conversation (default: 20)"
    )
    arg_parser.add_argument(
        "-s", "--scenario",
        help="Custom scenario description (overrides built-in scenarios)"
    )
    arg_parser.add_argument(
        "-o", "--output-dir", default=".",
        help="Output directory for generated files (default: current directory)"
    )
    args = arg_parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    scenarios = [args.scenario] * args.num_chats if args.scenario else SCENARIOS[:args.num_chats]

    for i, scenario in enumerate(scenarios):
        conversation = generate_conversation(client, scenario, args.turns)

        output_path = f"{args.output_dir}/generated_chat_{i + 1}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
