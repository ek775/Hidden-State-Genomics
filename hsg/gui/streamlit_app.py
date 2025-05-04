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

    # Set the page title and layout
    st.set_page_config(page_title="HSG Embeddings Dashboard", layout="wide")

    # Set the title of the app
    st.title("HSG Embeddings Dashboard")

    # Add a description or instructions for the app
    st.write("This is a dashboard for visualizing HSG embeddings from NCBI regulatory element tracks.")

    # Add your app logic here
    st.write("Hello, world!")

if __name__ == "__main__":
    main()