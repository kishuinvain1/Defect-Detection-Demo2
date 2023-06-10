import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64
from ultralytics import YOLO


@st.cache_resource(ttl=86400)
def load_model():
    model = YOLO('https://github.com/kishuinvain1/Defect-Detection-Demo2/blob/main/best.pt')
    return model


def load_image():
    opencv_image = None 
    path = None
    f = None
    name = None
    image_data = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read())).astype(np.uint8)
       
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        #cv2.imwrite("main_image.jpg", opencv_image)
       
    return image_data, opencv_image

	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-15)
    if(cl == "ok"):
        color = (0,255,0)
    elif(cl == "tear"):
        color = (255,0,0)
	
    #cl_cf = cl+""+str(cf)	
    
    img = cv2.rectangle(img, start_pnt, end_pnt, color, 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 10, cv2.LINE_AA)	
    st.image(img, caption='Resulting Image')	
    
	
def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Defect Detection V2')
    image, svd_img = load_image()
    model = load_model()
    result = st.button('Predict')
    if(result):
        st.write('Calculating results...')
        results = model("main_image.jpg", agnostic_nms=True)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No object is detected")
        else:
            conf_lst = results[0].conf.tolist()
            cls_lst = results[0].cls.tolist()
            bb_lst = results[0].xywh.tolist()
            x = bb_lst[0][0]
            y = bb_lst[0][1]	
            w = bb_lst[0][2]
            h = bb_lst[0][3]
            cl = cls_lst[0]
            cnf = conf_lst[0]   
            drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
        
              

if __name__ == '__main__':
    main()
