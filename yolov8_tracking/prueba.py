from ultralytics import YOLO
import cv2
import os
import numpy as np
#import pyttsx3
raiz = os.getcwd()
dir = os.path.join(raiz, "fotos")
dir2 = os.path.join(raiz, "fotos2")

model = YOLO("espalda.pt")

for folder in os.listdir(dir):
    if not os.path.isfile(os.path.join(dir, folder)):
        idx = 0
        for file in os.listdir(os.path.join(dir, folder)):
            if os.path.isfile(os.path.join(os.path.join(dir, folder),file)):
                idx = idx + 1
                os.makedirs('fotos2/'+folder+"/persona", exist_ok=True)
                os.makedirs('fotos2/'+folder+"/persona_frente", exist_ok=True)
                os.makedirs('fotos2/'+folder+"/persona_atras", exist_ok=True)
                im_dir = os.path.join(os.path.join(dir, folder),file)
                img = cv2.imread(im_dir)
                results = model.predict(source=im_dir, save=True, save_txt=True)

                if len(results[0].boxes.cls.numpy()) > 0:
                    if int(results[0].boxes.cls.numpy()[0]) == 0:
                        cv2.imwrite(os.path.join(os.path.join(os.path.join(dir2, folder),"persona"),"foto"+str(idx)+".png").replace('\\',"/") ,img)
                    elif int(results[0].boxes.cls.numpy()[0]) == 1:
                        cv2.imwrite(os.path.join(os.path.join(os.path.join(dir2, folder),"persona_frente"),"foto"+str(idx)+".png").replace('\\',"/") ,img)
                    elif int(results[0].boxes.cls.numpy()[0]) == 2:
                        cv2.imwrite(os.path.join(os.path.join(os.path.join(dir2, folder),"persona_atras"),"foto"+str(idx)+".png").replace('\\',"/") ,img)
#for r in results:
#    boxes = r.boxes
#    for box in boxes:
#        print(box)
#engine = pyttsx3.init()
#engine.say("Hola Carlos Revelo esto es una prueba de texto")
#engine.runAndWait()




