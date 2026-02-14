import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
model = "claude-sonnet-4-20250514"
chunk_size = 10
segment_window_size = 80
segment_overlap = 10
