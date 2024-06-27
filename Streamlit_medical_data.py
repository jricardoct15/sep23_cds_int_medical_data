import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#Plot grayscale Image function
def PlotXRay (Image):
  plt.figure(figsize = (8,5))
  if len(Image.shape) == 2 :
    plt.imshow(Image, cmap = 'gray')
  else:
    plt.imshow(Image, cmap = 'RGB')
  plt.xticks([])
  plt.yticks([])

  plt.show();

st.title("Lung X-Ray : classification project")

st.write("### Load an X-Ray JPG image to classify")

uploaded_file = st.file_uploader("Choose a file", type='jpg')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    file = uploaded_file.read() # Read the data
    st.write("The image ", uploaded_file.name, "was load successufully, with ",uploaded_file.size," Bytes.")
    st.image(uploaded_file)
    PlotXRay(cv2.imread(uploaded_file,format='jpg'))
    