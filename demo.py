import streamlit as st
from streamlit_option_menu import *
import cv2
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os



FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h,w)= frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

with st.sidebar:
    choose = option_menu("Menu", ["Mask Detector", "About", "Project Report", "Project code", "Contact"],
                         icons=['tablet', 'activity', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#2e3047"},
        "icon": {"color": "white", "font-size": "20px"},
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#FFFF00"},
    }
    )
#"nav-link-selected": {"background-color": "#FFFF00"},
if choose=="Mask Detector":
    #original_title = '<p style="font-family:Rockwell; color:black; font-size: 45px;">Real Time Face Mask Detector</p>'
    #st.markdown(original_title, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Real Time Face Mask Detector</p>', unsafe_allow_html=True)
    #st.markdown("Real Time Face Mask Detector")
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    except Exception:
        st.write("Error loading cascade classifiers")
    def get_cap():
        return cv2.VideoCapture(0)
    cap = get_cap()
    frameST = st.empty()
    while True:
        ret, frame = cap.read()
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_ITALIC, 1, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # cv2.imshow("FRAME_WINDOW", frame)

        frameST.image(frame, channels="BGR")

elif choose == "About":
    st.header("About")
    st.markdown("Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus")
    st.markdown("Most people infected wite the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illnesses. Anyone can get sick with COVID-19 and become seriously ill or die at any age.Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illnesses. Anyone can get sick with COVID-19 and become seriously ill or die at any age.")
    st.image("face_mask.png")
    st.markdown("The project's Real-time face mask detector helps to detect masks in real-time. By taking the video as output it detects whether the person is wearing a mask or not.")


elif choose == "Project Report":
    st.header("Project Report")
    st.markdown("The project face mask detector detects whether the person is wearing a mask or not in real-time. The application is divided into 3 phases")
    st.subheader("1.Data Preprocessing")
    st.image("pc1.png")
    st.markdown("The data preparation is an important part Of the whole process.  The images will be loaded from the disk with the labels")
    st.markdown("The images will be resizes into the size (224,224) and coverted into arrays.")
    st.markdown("The labels are also converted into numerical using the label binarizer.")
    st.image("DP.png")
    st.subheader("2.Training the Model")
    st.image("pc2.png")
    st.markdown("Model training is the phase in the data science development lifecycle where practitioners try to fit the best combination of weights and bias to a machine-learning algorithm to minimize a loss function over the prediction range. The purpose of model training is to build the best mathematical representation of the relationship between data features and a target label or among the features themselves.")
    st.markdown("For training the above model, two data sets are used one is labeled with persons with masks and the other is labeled with the person without the mask.")
    st.image("dp2.png")
    st.subheader("3.Integrating the model with streamlit framework")
    st.image("pc3.png")
    st.markdown("After building and training the model the next step is integrating with the streamlit.")
    st.markdown("Streamlit is an open-source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib, etc.")
    st.markdown("The model is loaded and uses streamlit components to detect the person's face and predict whether the area wearing a mask or not.")
    st.image("st2.png")
elif choose == "Project code":
    st.header("Project Code")
    st.markdown("To get the data set of images: ")
    st.markdown("https://github.com/skshareef/datset_face_mask.git")
    st.markdown("The source code is available at : ")
    st.markdown("https://github.com/skshareef")
    st.markdown("To learn more about face and mask detection:")
    st.markdown("https://iopscience.iop.org/article/10.1088/1742-6596/1916/1/012084")
    st.markdown("To learn more about the streamlit:")
    st.markdown("https://docs.streamlit.io/library/api-reference")
    st.markdown("To learn more about the OpenCV: ")
    st.markdown("https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html")
    st.markdown("To learn more about the tensor flow:")
    st.markdown("https://www.tensorflow.org/api_docs")
    st.markdown("To learn more about the Keras;")
    st.markdown("https://keras.io/api/")
    st.markdown("To learn more about the mobile net:")
    st.markdown("https://arxiv.org/abs/1704.04861")

elif choose == "Contact":
    st.header("CONTACT PAGE")
    st.markdown("The web application is developed as a capstone project by the students of the lovely professional university.")
    st.subheader("For any queries ")
    st.markdown("Contact:vignayalamarthi23@gmail.com")
    st.markdown("shaikshareef7537@gmail.com")
