import numpy as np
import cv2
from keras.preprocessing import image
import face_recognition
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import mysql.connector
import sqlite3
import marshal


# _______________ connecting to database _________________
conn = sqlite3.connect('customers.sqlite')
cursor = conn.cursor()
print('Opened Successfully')
# ________________________________________________________

# __________________ delete function ___________________________
def delete_from_known_faces(match,known_face_encodings):
    list_size = len(match)
    i=0
    idx=0
    for i in range(0,list_size,1):
        if match[i] == True:
            idx = i
            break

    known_face_encodings.pop(idx)

# __________________ get_unique_id ___________________________
def get_unique_id(match,known_face_encodings):
    list_size = len(match)
    i=0
    idx=0
    for i in range(0,list_size,1):
        if match[i] == True:
            idx = i
            break

    return idx

# __________________ check function ___________________________
def check(match):
    for element in match:
        if element == True:
            return True
    return False
# _____________________________________________________________


#opencv initialization

# face_cascade = cv2.CascadeClassifier(r'C:\Users\Diwakar Singh\Documents\Face-Recognition-System-SDL-Project\Expression Recognition\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(r'C:\Users\Diwakar Singh\Documents\Face-Recognition-System-SDL-Project\SharpView CCTV in Shopping Mall.mp4')
cap_exit = cv2.VideoCapture(0)

#----------------------------- 
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open(r"C:\Users\Diwakar Singh\Documents\Face-Recognition-System-SDL-Project\Expression Recognition\facial_expression_model_structure.json", "r").read())
model.load_weights(r'C:\Users\Diwakar Singh\Documents\Face-Recognition-System-SDL-Project\Expression Recognition\facial_expression_model_weights.h5') #load weights

#_______________________________
detector = MTCNN()
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

d = face_recognition.load_image_file(r'C:\Users\Diwakar Singh\Documents\Face-Recognition-System-SDL-Project\FaceDetector\IMG20190827113940 (2).jpg')
face_encoding_diwakar = face_recognition.face_encodings(d)[0]

known_face_encodings = [face_encoding_diwakar]
exit_known_face_encodings = []

while(True):
    ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = img[:, :, ::-1]
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces = detector.detect_faces(img)

	#print(faces) #locations of detected faces
    
    # face_encodings = face_recognition.face_encodings(img)
    # for face_encoding in face_encodings:
    #     match = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)
    #     if check(match) == False:
    #         # print('new face detected')
    #         known_face_encodings.append(face_encoding)

        
    for result in faces:
        x,y,w,h = result['box']
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        face_encodings = face_recognition.face_encodings(detected_face)
        
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            match = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)
            if check(match) == False:
                print('new face detected')
                known_face_encodings.append(face_encoding)
                
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral 
                max_index = np.argmax(predictions[0])
                emotion = emotions[max_index]
                val = predictions[0][3]
                # print(val)
                data = marshal.dumps(face_encoding)
                # print(data)
                cursor.execute('''insert into customer (Id,Entry_happiness) values (?,?)''',(data,val))
                # print(emotion)
		
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral 
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]

		#write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#process on detected face end
		#-------------------------

    cv2.imshow('img',img)
# ______________________________________________________________________________________________________
    ret_exit, img_exit= cap_exit.read()
    gray = cv2.cvtColor(img_exit, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = img_exit[:, :, ::-1]
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces = detector.detect_faces(img_exit)

	#print(faces) #locations of detected faces
    
    # face_encodings = face_recognition.face_encodings(img)
    # for face_encoding in face_encodings:
    #     match = face_recognition.compare_faces(exit_known_face_encodings,face_encoding,tolerance=0.5)
    #     if check(match) == False:
    #         match_for_entry = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)
    #         if check(match_for_entry) == True:
    #             # print('new face detected')
    #             unique_id = get_unique_id(match_for_entry,known_face_encodings)
    #             delete_from_known_faces(match_for_entry,known_face_encodings)
    #             exit_known_face_encodings.append(unique_id)

        
    for result in faces:
        x_exit,y_exit,w_exit,h_exit = result['box']
        cv2.rectangle(img,(x_exit,y_exit),(x_exit+w_exit,y_exit+h_exit),(255,0,0),2) #draw rectangle to main image
        
        detected_face_exit = img_exit[int(y_exit):int(y_exit+h_exit), int(x_exit):int(x_exit+w_exit)] #crop detected face
        face_encodings_exit = face_recognition.face_encodings(detected_face_exit)
        detected_face_exit = cv2.cvtColor(detected_face_exit, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face_exit = cv2.resize(detected_face_exit, (48, 48)) #resize to 48x48
        
        if len(face_encodings_exit) > 0:
            face_encoding_exit = face_encodings_exit[0]
            match = face_recognition.compare_faces(exit_known_face_encodings,face_encoding_exit,tolerance=0.5)
            if check(match) == False:
                match_for_entry = face_recognition.compare_faces(known_face_encodings,face_encoding_exit,tolerance=0.5)
                if check(match_for_entry) == True:
                    # print('new face detected')
                    idx = get_unique_id(match_for_entry,known_face_encodings)
                    unique_id = known_face_encodings[idx]
                    delete_from_known_faces(match_for_entry,known_face_encodings)
                    exit_known_face_encodings.append(unique_id)
                    
                    img_pixels_exit = image.img_to_array(detected_face_exit)
                    img_pixels_exit = np.expand_dims(img_pixels_exit, axis = 0)
                    img_pixels_exit /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                    predictions_exit = model.predict(img_pixels_exit) #store probabilities of 7 expressions
                    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral 
                    max_index_exit = np.argmax(predictions_exit[0])
                    emotion_exit = emotions[max_index_exit]
                    # print(emotion_exit)
		
        img_pixels_exit = image.img_to_array(detected_face_exit)
        img_pixels_exit = np.expand_dims(img_pixels_exit, axis = 0)
		
        img_pixels_exit /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions_exit = model.predict(img_pixels_exit) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral 
        max_index_exit = np.argmax(predictions_exit[0])
        emotion_exit = emotions[max_index]

		#write emotion text above rectangle
        cv2.putText(img_exit, emotion_exit, (int(x_exit), int(y_exit)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		#process on detected face end
		#-------------------------

    cv2.imshow('img_exit',img_exit)
# ____________________________________
# _______________________________________________________________________________________________________

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break
#kill open cv things
print(len(known_face_encodings))		
cap.release()
cv2.destroyAllWindows()
