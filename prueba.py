import cv2
import os
import numpy as np
import face_recognition
import time

#import pyttsx3
raiz = os.getcwd()
dir = os.path.join(raiz, "fotos")
dir2 = os.path.join(raiz, "fotos2")

for folder in os.listdir(dir):
    if not os.path.isfile(os.path.join(dir, folder)):
        idx = 0
        idx2 = 0
        idx3 = 0
        for file in os.listdir(os.path.join(dir, folder)):
            
            if os.path.isfile(os.path.join(os.path.join(dir, folder),file)):
                
                os.makedirs('fotos2/'+folder+"/persona", exist_ok=True)
                os.makedirs('fotos2/'+folder+"/persona_frente", exist_ok=True)
                im_dir = os.path.join(os.path.join(dir, folder),file)
                img = cv2.imread(im_dir)
                #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                #img_width, img_height = 112, 92
                #img = cv2.flip(img, 1, 0)
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                inicio = time.time()
                face_loc = face_recognition.face_locations(img)
                fin = time.time()
                print(fin-inicio)
                #if type(faces) is not tuple:
                if face_loc != []:
                    ( y, w, h, x) = face_loc[0]

                    idx = idx + 1
                    idx2 = idx2 + 1
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dir2, folder),"persona_frente"),"foto"+str(idx)+".png").replace('\\',"/") ,img)
                    os.makedirs('fotos2/'+folder+"/persona_frente/cara", exist_ok=True)
                    #(x, y, w, h) = faces[0]
                    ## Con cabello
                    #img_recortada = img[y:y+h, x: x+w]  
                    ##Sin cabello
                    #img_recortada = img[y:y+h, x: x+w]
                    img_recortada = img[y:h, x:w]
                    
                    #img_recortada = img[face_loc[3]:y+h, face_loc[2]: x+w]

                    cv2.imwrite(os.path.join(os.path.join(os.path.join(os.path.join(dir2, folder),"persona_frente"),"cara"),"cara"+str(idx2)+".png") ,img_recortada)
                    #Dibujamos un rectangulo en las coordenadas del rostro
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    #cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 3)
                    #Ponemos el nombre en el rectagulo
                    #cv2.putText(img, folder, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  
                    #cv2.imshow('Detected faces', img)
                    #cv2.waitKey(0)
                            
                    ##for (x, y, w, h) in faces:
                        ##cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
                    ##results = model.predict(source=im_dir)
                    ##, save=True, save_txt=True
                else:
                    idx3 = idx3 + 1
                    cv2.imwrite(os.path.join(os.path.join(os.path.join(dir2, folder),"persona"),"foto"+str(idx3)+".png").replace('\\',"/") ,img)

              
#    for box in boxes:
#        print(box)
#engine = pyttsx3.init()
#engine.say("Hola Carlos Revelo esto es una prueba de texto")
#engine.runAndWait()




