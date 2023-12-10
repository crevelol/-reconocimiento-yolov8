import os
import face_recognition
import cv2
import numpy as np
import shutil

cont = 0
raiz = os.getcwd()
dir = os.path.join(raiz, "fotos2")
dir2 = os.path.join(raiz, "fotos")

for folder in os.listdir(dir):
    if not os.path.isfile(os.path.join(dir, folder)):
        if len(os.listdir(os.path.join(dir, folder, 'persona_frente'))) > 1:
            for idx, file in enumerate(os.listdir(os.path.join(dir, folder, 'persona_frente'))):
                comparador = os.listdir(os.path.join(dir, folder, 'persona_frente'))
                if os.path.isfile(os.path.join(os.path.join(dir, folder, 'persona_frente'),file)) and idx > 1 and comparador[idx-1] != 'cara':
                    image1 = cv2.imread(os.path.join(os.path.join(dir, folder, 'persona_frente'),comparador[idx-1]))
                    image2 = cv2.imread(os.path.join(os.path.join(dir, folder, 'persona_frente'),file))
                    hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                    hist_img1[255, 255, 255] = 0 #ignore all white pixels
                    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                    hist_img2[255, 255, 255] = 0  #ignore all white pixels
                    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # Find the metric value
                    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
             
                    if round(metric_val, 2) < 0.45:
                        file_cara = file.replace('foto','cara')
                        os.remove(os.path.join(os.path.join(dir, folder, 'persona_frente'),file))
                        os.remove(os.path.join(os.path.join(dir, folder, 'persona_frente'),'cara',file_cara))
                        print(f"Se elimino por similitud baja de: ", round(metric_val, 2),"   "+file + " y "+file_cara)
        

coincidencias = []
for folder in os.listdir(dir):
    if not os.path.isfile(os.path.join(dir, folder)):
        #Carpetas de 1 person
        for folder2 in os.listdir(os.path.join(dir, folder)):
            if not os.path.isfile(os.path.join(os.path.join(dir, folder),folder2)):
                #Carpetas de persona y persona_frente
                for file in os.listdir(os.path.join(dir, folder, folder2)):
                    if not os.path.isfile(os.path.join(dir, folder,folder2, file)):
                        #Carpeta cara
                        if len(os.listdir(os.path.join(dir, folder, folder2,file))) > 1:
                            aleatorio = os.listdir(os.path.join(dir, folder, folder2,file))[1]
                        else:
                            aleatorio = os.listdir(os.path.join(dir, folder, folder2,file))[0]
                        #referencia = cv2.imread(os.path.join(dir, folder,folder2, file, aleatorio))
                        referencia = face_recognition.load_image_file(os.path.join(dir, folder,folder2, file, aleatorio))
                        face_encodings_ref = face_recognition.face_encodings(referencia, known_face_locations=[( 0, np.shape(referencia)[1], np.shape(referencia)[0], 0)])[0]
                        print(folder+"   "+aleatorio)
                        # Segunda pasada para verificar carpetas con persona #
                        ######################################################
                        vec = []
                        for folder_c in os.listdir(dir):
                            cont = 0
                            val=0
                            if not os.path.isfile(os.path.join(dir, folder_c)):
                                #Carpetas de 1 person
                                for folder2_c in os.listdir(os.path.join(dir, folder_c)):
                                    if not os.path.isfile(os.path.join(os.path.join(dir, folder_c),folder2_c)):
                                        #Carpetas de persona y persona_frente
                                        for file_c in os.listdir(os.path.join(dir, folder_c, folder2_c)):
                                            if not os.path.isfile(os.path.join(dir, folder_c,folder2_c, file_c)):
                                                #Fotos persona y persona frente
                                                for file2_c in os.listdir(os.path.join(dir, folder_c,folder2_c, file_c)):
                                                    if os.path.isfile(os.path.join(dir, folder_c,folder2_c, file_c, file2_c)):
                                                        
                                                        #Fotos cara
                                                        #img = cv2.imread(os.path.join(dir, folder_c,folder2_c, file_c, file2_c))
                                                        img = face_recognition.load_image_file(os.path.join(dir, folder_c,folder2_c, file_c, file2_c))
                                                        #cv2.imshow('Detected faces', img[0:np.shape(img)[0],0:np.shape(img)[1]])
                                                        #cv2.waitKey(0)
                                                        #Codigo de cara
                                                        face_encodings = face_recognition.face_encodings(img, known_face_locations=[( 0, np.shape(img)[1], np.shape(img)[0], 0)])[0]
                                                        #Comparacion de rostros
                                                        result = face_recognition.compare_faces([face_encodings_ref], face_encodings)
                                                        if result[0] == True:
                                                            cont = cont + 1
                                                        #Dato minimo para unir carpetas de fotos
                                                        val = int(len(os.listdir(os.path.join(dir, folder_c,folder2_c, file_c)))*0.6)
                                                        if len(os.listdir(os.path.join(dir, folder_c,folder2_c, file_c))) == 1:
                                                            val = 1

                                if cont >= val and val != 0:
                                    vec.append(folder_c)
                                    
                        if len(vec) > 0:
                            coincidencias.append(np.sort(vec))

conjunto_coincidencias = set()

for elemento in coincidencias:
    if len(elemento) > 1:
        # Si el elemento es una lista, convertirlo a una tupla para poder ser añadido al conjunto
        conjunto_coincidencias.add(tuple(elemento))
    else:
        conjunto_coincidencias.add(elemento[0])

# Convertir el conjunto nuevamente a una lista
coincidencias_sin_duplicados = list(conjunto_coincidencias)

b = []
for i in coincidencias_sin_duplicados:
    if isinstance(i, tuple):
        tu = []
        for j in i:
            if j not in coincidencias_sin_duplicados:
                tu.append(j)
        if len(tu) == 1: 
            b.append(tuple(tu)[0])
        elif len(tu) != 0:
            b.append(tuple(tu))
    else:
        b.append(i)

for idx, i in enumerate(b):
    os.makedirs(f"datasets generados/persona {idx+1}", exist_ok=True)
    if isinstance(i, tuple):
        for idx2, j in enumerate(i):
            shutil.move(os.path.join(dir,j),os.path.join(raiz, "datasets generados", f'persona {idx+1}'))
            os.rename(os.path.join(raiz, "datasets generados", f'persona {idx+1}',j), os.path.join(raiz, "datasets generados", f'persona {idx+1}',f'{idx2+1}° observacion'))
    else:
        shutil.move(os.path.join(dir,i),os.path.join(raiz, "datasets generados", f'persona {idx+1}'))
        os.rename(os.path.join(raiz, "datasets generados", f'persona {idx+1}',i),os.path.join(raiz, "datasets generados", f'persona {idx+1}','1° observacion'))

shutil.rmtree(dir)
shutil.rmtree(dir2)