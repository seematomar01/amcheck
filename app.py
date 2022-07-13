from flask import Flask, render_template, Response
import cv2
import cv2
import numpy as np
import face_recognition as face_rec

import os
from datetime import  datetime
from flask import Flask, render_template, request, redirect, flash, url_for

import urllib.request

from werkzeug.utils import secure_filename

import os
path = 'employee images'
employeeImg = []
employeeName = []
myList = os.listdir(path)

from flask import Flask

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.secret_key = "secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)



def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('welcome to seasia' + name)

for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    employeeImg.append(curimg)
    employeeName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(employeeImg)


def getPrediction(filename):
    print(filename)

    
    #frame = load_img('uploads/'+filename, target_size=(224, 224))
    frame = face_rec.load_image_file('uploads/'+filename)

    #frame = img_to_array(frame)


    facesInFrame = face_rec.face_locations(frame)
    encodeFacesInFrame = face_rec.face_encodings(frame, facesInFrame)
        

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        if min(facedis) < 0.5:
            matchIndex = np.argmin(facedis)

            print(matchIndex)


            name = employeeName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #MarkAttendence(name)
            return name,min(facedis)

    return 'no match','inf'





@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            getPrediction(filename)
            label, acc = getPrediction(filename)
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()








