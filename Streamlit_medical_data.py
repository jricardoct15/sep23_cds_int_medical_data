import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, Flatten, Dense, BatchNormalization)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Plot grayscale Image function
def PlotXRay(Image):
    plt.figure(figsize=(8, 5))
    if len(Image.shape) == 2:
        plt.imshow(Image, cmap='gray')
    else:
        plt.imshow(Image, cmap='RGB')
    plt.xticks([])
    plt.yticks([])
    st.pyplot(plt)
# Streamlit app structure
st.title("X-Ray Image Classification")
# File upload and selection options
uploaded_files = st.file_uploader("Upload X-Ray metadata files", accept_multiple_files=True, type=["xlsx"])
x = st.radio('Select data subset', options=['All images', 'Subset balanced', 'Use augmented dataset'])
# Load all Excel files into one dataframe
if uploaded_files:
    df_COVID = pd.read_excel(uploaded_files[0])
    df_Lung_Opacity = pd.read_excel(uploaded_files[1])
    df_Normal = pd.read_excel(uploaded_files[2])
    df_Pneumonia = pd.read_excel(uploaded_files[3])
    # Process the data
    df_COVID = df_COVID.drop(['FORMAT', 'SIZE', 'URL'], axis=1)
    df_Lung_Opacity = df_Lung_Opacity.drop(['FORMAT', 'SIZE', 'URL'], axis=1)
    df_Normal = df_Normal.drop(['FORMAT', 'SIZE', 'URL'], axis=1)
    df_Pneumonia = df_Pneumonia.drop(['FORMAT', 'SIZE', 'URL'], axis=1)
    df_COVID['Tag'] = "COVID"
    df_Lung_Opacity['Tag'] = "Lung_Opacity"
    df_Normal['Tag'] = "Normal"
    df_Pneumonia['Tag'] = "Viral Pneumonia"
    # Correct the Normal FILE NAME
    df_Normal['FILE NAME'] = df_Normal[df_Normal['Tag'].str.contains("Normal")]['FILE NAME'].str.replace('NORMAL', 'Normal')
    XRay = pd.concat([df_COVID, df_Lung_Opacity, df_Normal, df_Pneumonia])
    st.write(XRay.shape)
    st.write(XRay.head())
    # Load image and mask example
    BasePath = '/content/drive/MyDrive/Colab Notebooks/X_Ray_Project/'
    index = 87
    ImgPathFile = BasePath + XRay.iloc[index]['Tag'] + '/images/' + XRay.iloc[index]['FILE NAME'] + '.png'
    imgX = mpimg.imread(ImgPathFile)
    MaskPathFile = BasePath + XRay.iloc[index]['Tag'] + '/masks/' + XRay.iloc[index]['FILE NAME'] + '.png'
    maskX = mpimg.imread(MaskPathFile)
    GrayMaskX = maskX[:, :, 0]
    # Check image and mask size and formats
    st.write(f"imgX shape: {imgX.shape}, dtype: {imgX.dtype}, min: {imgX.min()}, max: {imgX.max()}")
    st.write(f"maskX shape: {maskX.shape}, dtype: {maskX.dtype}, min: {maskX.min()}, max: {maskX.max()}")
    st.write(f"GrayMaskX shape: {GrayMaskX.shape}, dtype: {GrayMaskX.dtype}, min: {GrayMaskX.min()}, max: {GrayMaskX.max()}")
    # Calculate cutImage TEST
    GrayMaskX_resized = cv2.resize(GrayMaskX, (imgX.shape[1], imgX.shape[0]))
    cutImage = imgX * GrayMaskX_resized
    PlotXRay(cutImage * 255)
    # Example: define and compile model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=[224, 224, 3]),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary(print_fn=lambda x: st.text(x))
    # Example: train model
    history = model.fit(train_generator,
    validation_data=valid_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    epochs=2,)
    st.write("Training history", history.history)