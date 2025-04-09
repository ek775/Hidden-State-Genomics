from build_SQL import GenBankDatabaseHandler
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Example transcript IDs to process
transcript_ids = [
    "NM_000277.2",  # Example ID
    "NM_001301717.2",  # Another ID
]

# Create the database handler (uses .env path if not passed)
db_handler = GenBankDatabaseHandler(email="your_email@example.com")

for tid in transcript_ids:
    db_handler.process_transcript(tid)