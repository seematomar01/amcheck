import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
s = face_rec.load_image_file('seema.jpg')
s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
s = resize(s, 0.50)
s_test = face_rec.load_image_file('akash.jpg')
s_test = resize(s_test, 0.50)
s_test = cv2.cvtColor(s_test, cv2.COLOR_BGR2RGB)

# finding face __cpLocation

faceLocation_s = face_rec.face_locations(s)[0]
encode_s = face_rec.face_encodings(s)[0]
cv2.rectangle(s, (faceLocation_s[3], faceLocation_s[0]), (faceLocation_s[1], faceLocation_s[2]), (255, 0, 255), 3)


faceLocation_stest = face_rec.face_locations(s_test)[0]
encode_stest = face_rec.face_encodings(s_test)[0]
cv2.rectangle(s_test, (faceLocation_s[3], faceLocation_s[0]), (faceLocation_s[1], faceLocation_s[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_s], encode_stest)
print(results)
cv2.putText(s_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', s)
cv2.imshow('test_img', s_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
