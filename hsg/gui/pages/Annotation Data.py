import streamlit as st
from hsg.gui.visualizations import CloudDataHandler, full_track_feat_avg

# configure title and layout
st.title("Annotation Data Explorer")
st.write("This is an annotation data explorer for HSG embeddings from NCBI regulatory element tracks.")
st.write("Use the sidebar to select annotation data and explore their embeddings.")
# Add a sidebar for annotation data selection
st.sidebar.title("Annotation Data Selection")
st.sidebar.write("Select annotation data to explore their embeddings.")


# initialize objects
data_handle = CloudDataHandler()  # should be cached by streamlit

# add dropdown menus to sidebar
# viewer selection
with st.sidebar:
    # Choose expansion factor
    # TODO: generate data for all expansion factors
    expansion = st.selectbox("Expansion Size", [8])
    # Choose layer
    layer = st.selectbox("layer", [i for i in range(24)])
    # Choose track
    track = st.selectbox("NCBI Regulatory Element Track", st.session_state.tracks)

# Add a button to retrieve data
if st.sidebar.button("GO!", type="primary"):
    # get selected track data
    with st.spinner("Getting Track Data..."):
        st.session_state.figureone = full_track_feat_avg(data_handle, 
                                  expansion=expansion, 
                                  layer=layer, 
                                  track=track)
        st.session_state.title = f"Annotation Data: ef{expansion} - layer {layer} - {track}"
        st.success("Data loaded successfully!")

# Display the main figure
if "figureone" in st.session_state:
    st.pyplot(st.session_state.figureone)