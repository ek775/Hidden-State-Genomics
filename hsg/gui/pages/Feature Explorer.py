import streamlit as st

# configure title and layout
st.title("Feature Explorer")
st.write("This is a feature explorer for HSG embeddings from NCBI regulatory element tracks.")
st.write("Use the sidebar to select features and explore their embeddings.")
# Add a sidebar for feature selection
st.sidebar.title("Feature Selection")
st.sidebar.write("Select features to explore their embeddings.")