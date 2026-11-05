import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cnn = tf.keras.models.load_model(
    'Emotion_detection_model.keras'
)
print('model loaded successfully')
trainint_file_name = "haarcascade_frontalface_default.xml"
model = cv2.CascadeClassifier(trainint_file_name)
emotion_label = ['Anger' , 'Fear' , 'Happy' , 'Neutral' , 'Sad']
while True:
    status , img = cam.read()
    if not status:
        print('camera is not working')
        break
    faces = model.detectMultiScale(img)
    for face in faces:
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cropped_image = img[y1:y2,x1:x2]
        # cropped_image = cv2.resize(cropped_image,(128,128))
        # cropped_image = cropped_image.reshape(1,128,128,3)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (128,128))
        cropped_image = cropped_image.astype('float32') / 255.0
        cropped_image = cropped_image.reshape(1,128,128,3)

        prediction = np.argmax(cnn.predict(cropped_image),axis=1)
        cv2.putText(img,emotion_label[prediction[0]],(x1,y1-10),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,[255,255,0],2)
    cv2.imshow('video..' , img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
cam.release()
        