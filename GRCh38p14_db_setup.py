import sqlite3
from Bio import SeqIO

# import data
print("Loading Data...")
GRCh38p14_refseq_path = './data/ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna'
seqrecords = []
with open(GRCh38p14_refseq_path, 'r') as file:
    for i, assembly in enumerate(SeqIO.parse(file, 'fasta')):
        seqrecords.append(assembly)
        if i % 10 == 0:
            print(f"{i} records read")

print("=== Data Loaded ===")

# create database connection
print("Connecting to Database...")
con = sqlite3.connect('./GRCh38p14.db')
cur = con.cursor()
print("=== Connected ===")

# check table exists
print("Creating table: GRCh38p14")
table_exists = False
try:
    res = cur.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type=table AND name=GRCh38p14;
                """)
    if res.fetchone() is None:
        table_exists = False
    else:
        table_exists = True
except:
    # proceed to create table
    pass

# create table
if table_exists == False:
    res = cur.execute("""
                CREATE TABLE GRCh38p14(id, description, seq)
                """)
    
    # load data
    for i in seqrecords:
        cur.execute
