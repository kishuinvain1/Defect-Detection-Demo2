import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image = None 
    path = None
    f = None
    name = None
    image_data = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        #data = Image.fromarray(bytearray(img_array))
        #data.save("main_image.jpg")
        
       
        #opencv_image = cv2.imdecode(img_array, 1)
        
	
        
       
    return img_array
       


	



	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    

    img = cv2.imread("main_image.jpg")
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    #img = saved_image.copy()
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
    
	
    #Model api for rubber part detection 2classes (Tear/Ok)
    rf = Roboflow(api_key="96lGMEBVBOSTljKY64Rp")
    project = rf.workspace().project("rubberpart-checking")
    model = project.version(1).model
    
     
    svd_img = load_image()
    print("saved images is...")
    print(svd_img)
    
    #st.write('Enter the image URL')
    #url = st.text_input('URL', '')
    result = st.button('Predict')
    if(result):
        st.write('Calculating results...')
        
	
	
	
	
	
	
        results = predict(model, svd_img)
	
        #results = predict(model2, "main_image.jpg")
        print("Prediction Results are...")	
        st.write(results['predictions'][0]['class'])
        st.write(results['predictions'][0]['confidence'])
	
        """
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No object is detected")
        else:
		
            new_img_pth = results['predictions'][0]['image_path']
            x = results['predictions'][0]['x']
            y = results['predictions'][0]['y']
            w = results['predictions'][0]['width']
            h = results['predictions'][0]['height']
            cl = results['predictions'][0]['class']
            cnf = results['predictions'][0]['confidence']
            
            
            drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
         """
              

if __name__ == '__main__':
    main()
