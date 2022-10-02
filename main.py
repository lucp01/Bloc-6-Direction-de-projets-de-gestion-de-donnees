import streamlit as st
import numpy as np
import os
import cv2
#Dont forget pip install opencv-python
import matplotlib.pyplot as plt
import tensorflow as tf
import mahotas
from PIL import Image
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler

# fix file size for feature extraction
fixed_size_ft_extract  = tuple((224, 224))
classes = {0:'drawings',1:'engraving',2:'iconography',3:'painting',4:'sculture'}

# features description 1:  Hu Moments (7 features)
# Nb. : les Hu momments sont basés sur l"analyse des contours
def fd_hu_moments(image):
    # convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the HuMoments feature 
    features = cv2.HuMoments(cv2.moments(image)).flatten()

    return features

# feature description 2: Haralick Texture (13 features)
# Nb. : La texture est caractérisée par la distribution spatiale des niveaux d'intensité dans un quartier.
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Compute the haralick texture feature 
    features = mahotas.features.haralick(gray).mean(axis=0)
    
    return features

# feature description 3: Color Histogram (512 features)
def fd_histogram(image, mask=None):
    # bins for histograms 
    bins = 8
    
    # conver the image to HSV colors-space
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #COMPUTE THE COLOR HISTOGRAM
    features  = cv2.calcHist([image2],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    
    # normalize the histogram
    cv2.normalize(features, features)
    
    return features.flatten()

# -------------------------------------------- 
# Predict image class with decision tree model
# -------------------------------------------- 
def predict_Trees(images_cv2, model_file):

    features=[]

    # load, no need to initialize the loaded_rf
    model = joblib.load(model_file)

    #Preprocess the image

    for image_cv2 in images_cv2:   
        # Compute features
        fv_hu_moments = fd_hu_moments(image_cv2)
        fv_haralick   = fd_haralick(image_cv2)
        fv_histogram  = fd_histogram(image_cv2)
            
        # Concatenate features
        features.append(np.hstack([fv_histogram, fv_haralick, fv_hu_moments]))
    
    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)
    
    # Make prédictions
    preds = model.predict(np.array(rescaled_features))
    #preds = model.predict(rescaled_features)
    #preds = model.predict(np.asarray(rescaled_features))
        
    # return prediction as well as class probabilities
    return ([str(preds[i])+'-'+classes[preds[i]] for i in range(len(preds))], preds)

# -------------------------------------------- 
# Predict image class with neural network FineTuningInceptionV3
# -------------------------------------------- 
def predict_FineTuningInceptionV3(images):

    # Load the model
    with open('2.0-FineTuningInceptionV3_model.sav', 'rb') as f:
        model = pickle.load(f)
    
    preds = []
    for image in images:  
        
        # Pre-process the images
        array_image = np.asarray(image)
        array_image = array_image/255
        array_image =  np.expand_dims(array_image,axis=0)
        
        # Make prédiction
        preds.append(model.predict([array_image]))

    # return prediction as well as class probabilities
    return ([str(np.argmax(preds[i]))+'-'+classes[np.argmax(preds[i])] for i in range(len(preds))], preds)

# -------------------------------------------- 
# Predict image class with neural network FineTuningMobileNet
# -------------------------------------------- 
def predict_FineTuningMobileNet(images):

    # Load the model
    model = tf.keras.models.load_model("2.1-FineTuningMobileNet.h5")
    
    # Pre-process the images
    preds = []
    for image in images:
        array_image = np.asarray(image)
        array_image = array_image/255
        array_image = np.expand_dims(array_image, axis=0)
        
        # Make prédiction
        preds.append(model.predict([array_image]))
    
    # return prediction as well as class probabilities
    return ([str(np.argmax(preds[i]))+'-'+classes[np.argmax(preds[i])] for i in range(len(preds))], preds)

# --------------------------------------------------
# Predict image class with several models to compare
# -------------------------------------------------- 
def predict(uploaded_files):
    
    if uploaded_files is not None and len(uploaded_files)>0:    
        
        # For each file uploaded, load and recize the image
        images=[]
        images_cv2=[]
        names=[]
        for file in uploaded_files:
            
            names.append(file.name)

            # open and resize image for all models (exept Trees models)
            image = Image.open(file)
            image = image.resize(fixed_size_ft_extract)
            images.append(image)
    
            # convert and resize image with cv2 (only for Trees models)
            image_cv2 = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
            image_cv2 = cv2.resize(image_cv2, fixed_size_ft_extract)
            images_cv2.append(image_cv2)
        
        # display the images
        st.image(images,caption=names)

        # Make prédictions
        st.text('Predictions :')
        st.text('=============')
        st.text("- DecisionTree : " + str(predict_Trees(images_cv2,"1.1-DecisionTree_final_model.joblib")[0]))
        st.text("- RandomForest : " + str(predict_Trees(images_cv2,"1.2-RandomForest_final_model.joblib")[0]))
        #st.text("- InceptionV3 : " + str(predict_FineTuningInceptionV3(images)[0]))
        st.text("- MobileNet : " + str(predict_FineTuningMobileNet(images)[0]))        
    
        # deleting uploaded saved picture after prediction

        # drawing graphs



#=========
#Main page
#=========
st.set_page_config(page_title="Smart_Art_Classification", layout="wide")
st.title('Smart Art Classifier')

#Creating Upload button, display uploaded image on the app, and call the predictor function which we had just created.
uploaded_files = st.file_uploader("Upload Image", accept_multiple_files=True)
predict(uploaded_files)
