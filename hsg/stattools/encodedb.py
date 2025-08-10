import requests, json
import time
from typing import Optional


class RESTClient:
    """
    Wrapper for ENCODE REST API. Simplifies interaction with this database for RAG annotation process.

    See https://www.encodeproject.org/help/rest-api/ for more information on the ENCODE REST API.
    """

    def __init__(self, base_url:str = "https://www.encodeproject.org"):
        self.base_url = base_url
        self.rate_limit = 10 # 10 requests per second

    def test_connection(self):
        """ Test the connection to the ENCODE REST API. """

        response = requests.get(self.base_url)

        return response.status_code, response.reason, response.elapsed.total_seconds()

    def search(self, query:str, type:Optional[str]=None, status:Optional[str]=None, file_format:Optional[str]=None, 
               assembly:Optional[str]=None, field:Optional[list[str]]=None, limit:Optional[int]=None, format:str = "json") -> dict:
        """
        Basic Search Functionality for the ENCODE REST API. Rate limit is enforced by the wrapper class.

        See https://www.encodeproject.org/help/rest-api/ for more information on the ENCODE REST API.

        Args:
            query: Search term to query the ENCODE database.
            type: Type of data to search for (e.g. Biosample, Experiment, File, etc.).
            status: Status of the data to search for (e.g. released, in progress, etc.).
            file_format: File format of the data to search for (e.g. fastq, bam, etc.).
            assembly: Genome assembly of the data to search for (e.g. GRCh38, mm10, etc.).
            field: List of fields to search for in the data.
            limit: Limit the number of search results returned.
            format: Format of the response from the ENCODE REST API (default is JSON
                    but can be changed to other formats like XML).

        Returns:
            JSON response from the ENCODE REST API, serialized as a python dictionary.
        """

        url = f"{self.base_url}/search/?"

        # construct query with optional arguments
        params = {}

        if type:
            params["type"] = type
        if status:
            params["status"] = status
        if file_format:
            params["file_format"] = file_format
        if assembly:
            params["assembly"] = assembly
        if field:
            params["field"] = field
        if limit:
            params["limit"] = limit

        params["format"] = format
        params["searchTerm"] = query

        # execute query
        response = requests.get(url, params=params)

        # rate limit (adds delay based on response time)
        if response.elapsed.total_seconds() < 1/self.rate_limit:
            time.sleep(1/self.rate_limit - response.elapsed.total_seconds())

        return response.json()
    

# Test the RESTClient
if __name__ == "__main__":
    client = RESTClient()
    print(f"Connection Test: {client.test_connection()}\n")

    print("Test Search Results:")
    search_results = client.search("skin", type="Biosample", limit=1)
    print(type(search_results))
    print(json.dumps(search_results, indent=4))