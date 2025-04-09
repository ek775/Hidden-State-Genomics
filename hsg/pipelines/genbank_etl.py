from Bio import Entrez, SeqIO

class GenBankFetcher:
    def __init__(self, transcript_id, email):
        """Initializes the fetcher with transcript ID and sets Entrez email."""
        self.transcript_id = transcript_id
        self.email = email
        Entrez.email = self.email
        self.record = None

    def fetch(self):
        """Fetches and stores the GenBank record."""
        try:
            handle = Entrez.efetch(db="nucleotide", id=self.transcript_id, rettype="gb", retmode="text")
            self.record = SeqIO.read(handle, "genbank")
            handle.close()
        except Exception as e:
            print(f"Error fetching {self.transcript_id}: {e}")
            self.record = None

    def extract_definition(self):
        """Extracts the DEFINITION field from the record."""
        if self.record:
            return self.record.description
        return None

    def extract_all_cds_info(self):
        """Extracts CDS features and translations."""
        if not self.record:
            return []

        cds_info = []
        for feature in self.record.features:
            if feature.type == "CDS":
                location = feature.location
                translation = feature.qualifiers.get("translation", ["N/A"])[0]
                gene_name = feature.qualifiers.get("gene", ["Unknown Gene"])[0]
                protein_id = feature.qualifiers.get("protein_id", ["No Protein ID"])[0]
                protein_name = feature.qualifiers.get("product", ["No Protein Name"])[0]
                cds_info.append({
                    "gene": gene_name,
                    "location": str(location),
                    "protein_id": protein_id,
                    "protein_name": protein_name,
                    "translation": translation
                })
        return cds_info

    def extract_origin_sequence(self):
        """Returns the full sequence (origin) of the record."""
        if self.record:
            return str(self.record.seq)
        return None