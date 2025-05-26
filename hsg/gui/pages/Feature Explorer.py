import streamlit as st
from hsg.gui.visualizations import CloudDataHandler
from hsg.gui.visualizations import feature_views


# configure title and layout
st.title("Regulatory Element Track-Feature Association Explorer")
st.write("This tool allows you to explore the association between SAE-derived features from NTv2 and known regulatory elements. Due to the limited context window " \
"each track is broken into 6kb fragments, with 50 bases included on the 5' and 3' ends of each contiguous track region.")
st.write("Use the sidebar to select features and explore their associations with known regulatory elements.")
st.divider()

# Add a sidebar for feature selection
st.sidebar.title("Feature Selection")
st.sidebar.write("Use the dropdown menus below to explore NTv2 feature associations.")


# initialize objects
data_handle = CloudDataHandler() # should be cached by streamlit

# load available tracks
if "tracks" not in st.session_state:
    with open("data/Annotation Data/tracks.txt", "r") as f:
        tracks = [line.strip() for line in f.readlines()]
        st.session_state.tracks = tracks

# viewer selection
with st.sidebar:
    # Choose expansion factor
    # TODO: generate data for all expansion factors
    expansion = st.selectbox("Expansion Size", [8])
    # Choose layer
    layer = st.selectbox("layer", [i for i in range(24)])
    # Choose track
    track = st.selectbox("NCBI Regulatory Element Track", st.session_state.tracks)
    # Choose fragment
    # TODO: show available fragments for the selected track
    fragment = st.selectbox(
        "Fragment", 
        data_handle.list_fragments(
            expansion=expansion, 
            layer=layer, 
            track=track
        ),
    )

# Add a button to retrieve data
if st.sidebar.button("GO!", type="primary"):
    # get selected track data
    with st.spinner("Loading data..."):
        # retrieve the data
        pearson_scores, xcorr_array = data_handle.retrieve_array(expansion=expansion, layer=layer, track=track, fragment=fragment)
        st.session_state.pearson_scores = pearson_scores
        st.session_state.xcorr_array = xcorr_array
        st.session_state.title = f"Feature Explorer: ef{expansion} - layer {layer} - {track} - window {fragment}"
        st.success("Data loaded successfully!")

    # Generate the feature views plot
    with st.spinner("Generating plots..."):
        st.session_state.figureone = feature_views(
            suptitle=st.session_state.title, 
            pearson_scores=st.session_state.pearson_scores, 
            xcorr=st.session_state.xcorr_array
        )

# Display main figure
if "figureone" in st.session_state:
    st.pyplot(st.session_state.figureone)
