import streamlit as st
import pandas as pd

st.title("Lung X-Ray : classification project")

st.write("### Load an X-Ray JPG image to classify")

uploaded_file = st.file_uploader("Choose a file", type='jpg')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

