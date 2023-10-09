from flask import Flask,render_template,redirect,request
import cv2
from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np
import csv
from datetime import datetime,date
import os

app=Flask(__name__)
name,ids='',''
json_file = open("./emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
if 'Faces' not in os.listdir():
    os.mkdir('Faces')

@app.route('/')
def Homepage():
    return render_template("main.html",namevar=name,idvar=ids)

@app.route('/add',methods=['POST'])
def Add():
    global name,ids
    name=request.form["input1"]
    ids=request.form["input2"]
    return redirect('/')

@app.route('/about')
def About():
    return render_template('about.html')

@app.route('/emo')
def Emo():
    model = model_from_json(model_json)

    model.load_weights("./my_model.h5")
    haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade=cv2.CascadeClassifier(haar_file)

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)
        return feature/255.0

    webcam=cv2.VideoCapture(0)
    labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

    while True:
        i,im=webcam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(im,1.3,5)
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            print("Predicted Output:", prediction_label)
            
            dateandtime=datetime.now().strftime("%d/%m/%y_%H:%M:%S")
            if f'{name}' not in os.listdir('Faces'):
                with open(f'Faces/{name}','w') as f:
                    f.write("Name,Time")    
            with open(f'Faces/{name}.csv','w') as f:
                f.writelines(f'\n{name},{dateandtime},{prediction_label}\n')

        cv2.imshow("Output",im)

        if cv2.waitKey()==ord('q'):
            break
    return redirect('/')

if __name__=="__main__":
    app.run(debug=True)