import os
import json
from dotenv import load_dotenv
import sqlite3
from genbank_etl import GenBankFetcher

# Load environment variables from .env file
load_dotenv()

class GenBankDatabaseHandler:
    def __init__(self, db_name=None, email="your_email@example.com"):
        load_dotenv()  # Load .env file
        self.db_name = os.path.expanduser(os.getenv("GENBANK_DB_PATH", "genbank_data.db"))
        self.email = email

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)

        self._initialize_database()

    def _initialize_database(self):
        """Initializes the SQLite database with a single unified table."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS GenBankEntries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_id TEXT UNIQUE,
                    description TEXT,
                    origin_sequence TEXT,
                    cds_info TEXT  -- JSON string storing list of CDS entries
                )
            ''')

            conn.commit()

    def is_transcript_in_db(self, transcript_id):
        """Checks if a transcript ID already exists in the database."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM GenBankEntries WHERE transcript_id = ?", (transcript_id,))
            return cursor.fetchone() is not None

    def store_in_database(self, transcript_id, description, origin_seq, cds_list):
        """Stores all GenBank data in a single row as a JSON blob."""
        cds_json = json.dumps(cds_list)  # Serialize CDS list

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO GenBankEntries (transcript_id, description, origin_sequence, cds_info)
                VALUES (?, ?, ?, ?)
            ''', (transcript_id, description, origin_seq, cds_json))
            conn.commit()

    def process_transcript(self, transcript_id):
        """Fetches, parses, and stores GenBank record for a given transcript ID."""
        if self.is_transcript_in_db(transcript_id):
            print(f"Transcript {transcript_id} already exists in the database. Skipping.")
            return

        fetcher = GenBankFetcher(transcript_id, self.email)
        fetcher.fetch()

        if not fetcher.record:
            return

        description = fetcher.extract_definition()
        origin_seq = fetcher.extract_origin_sequence()
        cds_list = fetcher.extract_all_cds_info()

        self.store_in_database(transcript_id, description, origin_seq, cds_list)
        print(f"Processed and stored: {transcript_id}")