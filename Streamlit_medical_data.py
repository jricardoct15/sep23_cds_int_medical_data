import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf

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
    PlotXRay(cv2.imread(uploaded_file))
    
    #Load the model
    model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/X_Ray_Project/Ricardo Model2.h5')
    st.write(model.summary())

    #predict
    st.write('### Prediction')
    predictonClass = model.predict(uploaded_file)
    st.write("The prediction for the image ", uploaded_file.name, "is ", predictonClass)

