#!/usr/bin/env python3
from typing import Any, Dict, Optional
import os, logging, sys, json, asyncio, random, time
from pathlib import Path
from enum import StrEnum
import fastapi
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# Check for required dependencies
missing_deps = []
try:
    import httpx
except ImportError:
    missing_deps.append("httpx")
try:
    from fastapi import HTTPException
except ImportError:
    missing_deps.append("fastapi")

if missing_deps:
    print("Error: Missing required dependencies. Please install them using:")
    print(f"pip install {' '.join(missing_deps)}")
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"

PUBLIC_URL = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"

BOLTZ_KEY = os.getenv("BOLTZ_KEY")
if not BOLTZ_KEY:
    logger.warning("BOLTZ_KEY not found in environment variables. Please set it in your .env file if executing outside of NGC.")



########################################################################################################


async def make_nvcf_call(function_url: str,
                        data: Dict[str, Any],
                        additional_headers: Optional[Dict[str, Any]] = None,
                        NVCF_POLL_SECONDS: int = 300,
                        MANUAL_TIMEOUT_SECONDS: int = 400) -> Dict:
    """
    Make a call to NVIDIA Cloud Functions using long-polling,
    which allows the request to patiently wait if there are many requests in the queue.
    """
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {BOLTZ_KEY}",
            "NVCF-POLL-SECONDS": f"{NVCF_POLL_SECONDS}",
            "Content-Type": "application/json"
            }
        if additional_headers is not None:
            headers.update(additional_headers)
        logger.debug(f"Headers: {dict(**{h: v for h, v  in headers.items() if 'Authorization' not in h})}")
        # TIMEOUT must be greater than NVCF-POLL-SECONDS
        logger.debug(f"Making NVCF call to {function_url}")
        logger.debug(f"Data: {data}")
        response = await client.post(function_url,
                                     json=data,
                                     headers=headers,
                                     timeout=MANUAL_TIMEOUT_SECONDS)
        logger.debug(f"NVCF response: {response.status_code, response.headers}")

        if response.status_code == 202:
            # Handle 202 Accepted response
            task_id = response.headers.get("nvcf-reqid")
            while True:
                ## Should return in 5 seconds, but we set a manual timeout in 10 just in case
                status_response = await client.get(STATUS_URL.format(task_id=task_id),
                                                   headers=headers,
                                                   timeout=MANUAL_TIMEOUT_SECONDS)
                if status_response.status_code == 200:
                    return status_response.status_code, status_response
                elif status_response.status_code in [400, 401, 404, 422, 500]:
                    raise HTTPException(status_response.status_code,
                                        "Error while waiting for function: ",
                                        response.text)
        elif response.status_code == 200:
            return response.status_code, response
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

########################################################################################################

class Polymer():
    def __init__(self, id: str, molecule_type: str, sequence: str):
        self.id = id
        self.molecule_type = molecule_type
        self.sequence = sequence

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "molecule_type": self.molecule_type,
            "sequence": self.sequence,
        }

class Ligand():
    def __init__(self, id: str, smiles: str, predict_affinity: bool = True):
        self.id = id
        self.smiles = smiles
        self.predict_affinity = predict_affinity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "smiles": self.smiles,
            "predict_affinity": self.predict_affinity
        }
    
class Payload():
    def __init__(self, polymers: list[Polymer], ligands: list[Ligand], recycling_steps: int = 1, 
                 sampling_steps: int = 50, diffusion_samples: int = 3, step_scale: float = 1.2, 
                 without_potentials: bool = True):
        self.polymers = polymers
        self.ligands = ligands
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.step_scale = step_scale
        self.without_potentials = without_potentials

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polymers": [polymer.to_dict() for polymer in self.polymers],
            "ligands": [ligand.to_dict() for ligand in self.ligands],
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "diffusion_samples": self.diffusion_samples,
            "step_scale": self.step_scale,
            "without_potentials": self.without_potentials
        }

########################################################################################################

async def boltz_predict(payload: Payload) -> tuple[int, Dict[str, Any]]:
    code, response = await make_nvcf_call(function_url=PUBLIC_URL,
                                    data=payload.to_dict())
    if code == 200:
        data = response.json()
        logger.debug(f"Boltz Prediction Successful: {code}")
        logger.info(f"Number of structures returned: {len(data['structures'])}")
        logger.info(f"Number of confidence scores: {len(data['confidence_scores'])}")
        if data['structures']:
            first_structure = data['structures'][0]
            logger.info(f"First structure format: {first_structure['format']}")
            logger.info(f"First structure length: {len(first_structure['structure'])} characters")
        return code, data
    if code == 429:
        logger.warning("=== Rate limit hit for Boltz-2 API ===")
        return code, response
    else:
        raise HTTPException(status_code=code, detail=response.text)

def make_polymers(sequences: list[str]) -> list[Polymer]:
    polymers = []
    for i, seq in enumerate(sequences):
        polymer = Polymer(id=f"A", molecule_type="rna", sequence=seq)
        polymers.append(polymer)
    return polymers

def make_ligands(structs: list[str]) -> list[Ligand]:
    ligands = []
    for i, smiles in enumerate(structs):
        ligand = Ligand(id=f"L{i+1}", smiles=smiles, predict_affinity=False)
        ligands.append(ligand)
    return ligands


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from hsg.sequence import transcribe

    parser = argparse.ArgumentParser(description="Generate RNA-Cisplatin complex structure using Boltz-2")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file containing RNA sequences")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save Boltz-2 predictions")
    
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    logger.debug(f"CSV columns: {df.columns.tolist()}")
    assert "Sequence (Quadratlas)" in df.columns, "CSV file must contain 'Sequence (Quadratlas)' column"
    assert "Identifier (Quadratlas)" in df.columns, "CSV file must contain 'Identifier (Quadratlas)' column"
    
    sequences = df["Sequence (Quadratlas)"].tolist()
    sequences = [transcribe(s) for s in sequences]  # convert to RNA sequences via reverse complement
    names = df["Identifier (Quadratlas)"].tolist()
    
    # try up to 2 cisplatin molecules per RNA
    for i, s in tqdm(enumerate(sequences), desc="Gathering Boltz-2 predictions"):
        quadratlas_id = names[i]
        for n in range(1, 3):

            # configure
            polymers = make_polymers([s])
            ligands = make_ligands(["[NH3][Pt]([NH3])(Cl)Cl" for _ in range(n)]) # explicit coordination cisplatin SMILES
            payload = Payload(polymers=polymers, ligands=ligands)
            
            # make the Boltz-2 prediction - due to rate limits, timeouts and retry limits are set
            done = False
            retry_delay = 10 # initial delay is small, unit = seconds
            retry_count = 0
            max_retries = 5

            # retry loop
            while not done:

                if retry_count > max_retries:
                    logger.error(f"Max retries exceeded for {quadratlas_id} with {n} cisplatin molecules. Skipping...")
                    break

                delta = 0
                try:
                    start = time.time()
                    code, data = asyncio.run(asyncio.wait_for(boltz_predict(payload), timeout=300)) # 5 minutes

                    # we hit the rate limit - wait and retry with exponential backoff
                    if code == 429:
                        logger.warning(f"Retrying Boltz-2 prediction due to rate limit in {retry_delay} seconds...")
                        retry_count += 1
                        retry_delay *= 2 # exponential backoff
                        time.sleep(retry_delay)
                        continue

                    end = time.time()
                    delta = end - start
                    done = True
                    
                except Exception as e:
                    logger.error(f"Error during Boltz-2 prediction: {e}")
                    break
                
                if delta < 30: # max limit is 40 rpm (allegedly)
                    delay = 30 - delta
                    logger.info(f"Waiting {delay:.2f} seconds before continuing to avoid hitting rate limits...")
                    time.sleep(delay)

            # Save the results to a JSON file
            output_path = Path(f"{args.outdir}/{quadratlas_id}_cisplatin_{n}.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved Boltz prediction to {output_path}")
