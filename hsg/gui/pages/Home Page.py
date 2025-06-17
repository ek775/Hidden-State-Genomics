import streamlit as st

# import environment variables
from dotenv import load_dotenv
load_dotenv()

# Set title and layout
st.title("HSG Embeddings Dashboard")

# initialize sidebar
with st.sidebar as bar:
    expansion = st.selectbox("Expansion Size", options=[8, 16, 32])
    layer = st.selectbox("Layer", options=[i for i in range(24)])

# initialize objects

if 'model' not in st.session_state:
    pass # TODO