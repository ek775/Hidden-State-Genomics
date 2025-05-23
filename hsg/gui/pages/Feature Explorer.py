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
data_handle = CloudDataHandler() # should be cached by streamlit

# load available tracks
with open("data/Annotation Data/tracks.txt", "r") as f:
    tracks = [line.strip() for line in f.readlines()]

# fetch data and render plots
with st.sidebar:
    # Choose expansion factor
    # TODO: generate data for all expansion factors
    expansion = st.selectbox("Expansion Size", [8])
    # Choose layer
    layer = st.selectbox("layer", [i for i in range(24)])
    # Choose track
    track = st.selectbox("NCBI Regulatory Element Track", tracks)
    # Choose fragment
    # TODO: show available fragments for the selected track
    fragment = st.selectbox("Fragment", [i for i in range(1, 2)])


if st.sidebar.button("GO!", type="primary"):
    # get selected track data
    with st.spinner("Loading data..."):
        # retrieve the data
        pearson_scores, xcorr_array = data_handle.retrieve_array(expansion=expansion, layer=layer, track=track, fragment=fragment)

    # Display the data
    with st.spinner("Generating plots..."):
        st.pyplot(plot_best_correlations(pearson_scores))
        st.pyplot(plot_all_feature_correlations(pearson_scores))
