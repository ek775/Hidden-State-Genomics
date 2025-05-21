import streamlit as st

# configure title and layout
st.title("Annotation Data Explorer")
st.write("This is an annotation data explorer for HSG embeddings from NCBI regulatory element tracks.")
st.write("Use the sidebar to select annotation data and explore their embeddings.")
# Add a sidebar for annotation data selection
st.sidebar.title("Annotation Data Selection")
st.sidebar.write("Select annotation data to explore their embeddings.")