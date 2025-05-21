import streamlit as st
from hsg.gui.visualizations import CloudDataHandler, Navigator
from hsg.gui.visualizations import plot_best_correlations, plot_all_feature_correlations


# configure title and layout
st.title("Feature Explorer")
st.write("This is a feature explorer for HSG embeddings from NCBI regulatory element tracks.")
st.write("Use the sidebar to select features and explore their associations with known regulatory elements.")


# Add a sidebar for feature selection
st.sidebar.title("Feature Selection")
st.sidebar.write("Use the dropdown menus below to explore NTv2 feature associations.")


# initialize objects
data_handle = CloudDataHandler()

# dropdown menu
with open("data/Annotation Data/tracks.txt", "r") as f:
    tracks = [line.strip() for line in f.readlines()]


# fetch data and render plots
if track := st.sidebar.selectbox("Select a feature", tracks, index=None):

    # get selected track data
    st.write(f"Selected track: {track}")
    st.write(f"Expansion: 8")
    st.write(f"Layer: 23")
    st.write(f"Fragment: 1")
    with st.spinner("Loading data..."):
        # retrieve the data
        pearson_scores, xcorr_array = data_handle.retrieve_array(expansion=8, layer=23, track=track, fragment=1)

    # Display the data
    with st.spinner("Generating plots..."):
        st.pyplot(plot_best_correlations(pearson_scores))
        st.pyplot(plot_all_feature_correlations(pearson_scores))
