# Script to create dashboard for HSG embeddings from NCBI regulatory element tracks

# imports
import streamlit as st

# data handling libraries
import numpy as np

# other utilities
from dotenv import load_dotenv

# built-in libraries
import os


### MAIN ###

def main():
    # Load environment variables from .env file
    load_dotenv()

    # load pages
    page_list = []
    pwd = os.path.dirname(os.path.abspath(__file__))

    for filename in os.listdir("hsg/gui/pages"):
        if filename.endswith(".py") and filename != "__init__.py":
            page_path = pwd + "/pages/" + filename
            page = st.Page(page=page_path, title=filename[:-3])
            page_list.append(page)

    current_page = st.navigation(pages=page_list)
    current_page.run()

if __name__ == "__main__":
    st.set_page_config(page_title="HSG Embeddings Dashboard", layout="wide")
    main()