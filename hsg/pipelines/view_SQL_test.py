import sqlite3
import os
from dotenv import load_dotenv

# Load DB path from .env
load_dotenv()
db_path = os.path.expanduser(os.getenv("GENBANK_DB_PATH", "genbank_data.db"))

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query entries where cds_info contains a specified gene named
print("=== GenBankEntries with gene ===")
query = """
SELECT *
FROM GenBankEntries
WHERE EXISTS (
    SELECT 1
    FROM json_each(GenBankEntries.cds_info)
    WHERE json_extract(json_each.value, '$.gene') IN (?, ?, ?)
)
"""
cursor.execute(query, ('PAH', 'CCR7', 'XYZ!@$'))
entries = cursor.fetchall()

for row in entries:
    print(row)

conn.close()