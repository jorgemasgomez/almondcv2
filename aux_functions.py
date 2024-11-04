import cv2
import numpy as np
import imutils
from sahi.slicing import slice_image
from PIL import Image
import os
import random
import shutil
import pandas as pd



def slicing(input_folder,output_directory,name_slicing,number_pictures,train_percen=60,val_percent=20,test_percent=20, slice_width=640, slice_height=640, overlap_height_ratio=0.2,
             overlap_width_ratio=0.2, crop="NA", crop_level=2):
    picture_list=os.listdir(input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    picture_list = [file for file in picture_list if file.lower().endswith(tuple(image_extensions))]
    if len(picture_list) < number_pictures:
        print("La carpeta no contiene suficientes imágenes.")
        return
    random_pictures = random.sample(picture_list, number_pictures)
    list_slices=[]

    output_folder=os.path.join(output_directory,name_slicing)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    # Verificar si la carpeta ya existe
    if os.path.exists(output_folder):
    # Si existe, eliminar todo el contenido
        shutil.rmtree(train_folder)
        shutil.rmtree(val_folder)
        shutil.rmtree(test_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    num_train = int(train_percen / 100 * number_pictures)
    num_val = int(val_percent / 100 * number_pictures)
    num_test = number_pictures - num_train - num_val

    # Crear subconjuntos de imágenes
    train_images = random_pictures[:num_train]
    val_images = random_pictures[num_train:num_train + num_val]
    test_images = random_pictures[num_train + num_val:]


    for image_input in random_pictures:
        image_path = os.path.join(input_folder, image_input)
        image_selected = Image.open(image_path)
        image_name, extension=os.path.splitext(image_input)
        #si la imagen es muy grande y tu objeto esta solo en la parte izquierda o derecha lo  puedes indicar y solo generara imagenes de esa parte de la imagen
        if crop=="left":
            width, height = image_selected.size
            image_selected = image_selected.crop((0, 0, width // crop_level, height))
        elif crop=="right":
            width, height = image_selected.size
            image_selected = image_selected.crop((width // crop_level, 0, width, height))
        
            # Determinar el subconjunto y carpeta de salida
        if image_input in train_images:
            output_subfolder = train_folder
        elif image_input in val_images:
            output_subfolder = val_folder
        else:
            output_subfolder = test_folder

        sliced=slice_image(image=image_selected, slice_width=slice_width, slice_height=slice_height, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio, 
                           output_dir=output_subfolder, verbose=True, output_file_name=f"SL_{image_name}")
        list_slices.append(sliced)
    return list_slices


#Habría que añadir a esta función una forma de ver las fotos si se quiere, con la mascara y el diametro, viendo el resultado si esta bien.
def obtain_pixel_metric(info_data, contours, output_directory, reference=24.25, smoothing=False,
                         smoothing_kernel=3, smooting_iterations=1):
    pixel_metric_list=[]

    for contour in contours:
        name_pic=os.path.basename(contour[1])
        mask=contour[0]
        if smoothing==True:
            # # # Aplicar operación de apertura para refinar los bordes del contorno
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smoothing_kernel, smoothing_kernel))
            mask=cv2.erode(mask, rect_kernel, iterations=smooting_iterations)
            mask=cv2.dilate(mask, rect_kernel, iterations=smooting_iterations)      
        
        mask_contours_list, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        i=0
        for contour_opencv in mask_contours_list:

            if len(mask_contours_list)>1:
                area=cv2.contourArea(contour_opencv)
                if i==0:
                    max_contour_area=contour_opencv
                    max_area=area
                elif area>max_area:
                    max_contour_area=contour_opencv
                i=i+1
            else:
                max_contour_area=contour_opencv
                pass
            # Calcular el diametro medio
        
        box= cv2.minAreaRect(max_contour_area)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        perimeter_reference= cv2.arcLength(box,True)
        average_diameter=(perimeter_reference)/4
        pixelsPerMetric = average_diameter / reference
        pixel_metric_list.append([name_pic,pixelsPerMetric])


    df_pix_met = pd.DataFrame(pixel_metric_list, columns=['Name_picture', 'Pixelmetric'])
    info_data_completed=pd.merge(info_data,df_pix_met,on='Name_picture')
    output=os.path.join(output_directory,"info_data_completed.txt")
    info_data_completed.to_csv(output, index=False, sep='\t')
    return info_data_completed


import os
import random
import shutil

def divide_in_sets(input_folder, output_directory, number_pictures,division_name, train_percent=60, val_percent=20, test_percent=20):
    # Obtener la lista de imágenes
    picture_list = os.listdir(input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    picture_list = [file for file in picture_list if file.lower().endswith(tuple(image_extensions))]
    
    # Verificar si hay suficientes imágenes
    if len(picture_list) < number_pictures:
        print("La carpeta no contiene suficientes imágenes.")
        return
    
    # Seleccionar un subconjunto de imágenes aleatoriamente
    random_pictures = random.sample(picture_list, number_pictures)
    
    # Crear las carpetas de salida (train, val, test)
    output_folder = os.path.join(output_directory, division_name)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Eliminar la carpeta si ya existe para evitar duplicados

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Determinar la cantidad de imágenes por conjunto
    num_train = int(train_percent / 100 * number_pictures)
    num_val = int(val_percent / 100 * number_pictures)
    num_test = number_pictures - num_train - num_val

    # Crear subconjuntos de imágenes
    train_images = random_pictures[:num_train]
    val_images = random_pictures[num_train:num_train + num_val]
    test_images = random_pictures[num_train + num_val:]

    # Mover las imágenes a sus respectivas carpetas
    for image in random_pictures:
        image_path = os.path.join(input_folder, image)
        if image in train_images:
            shutil.copy(image_path, train_folder)  # Copiar a la carpeta de entrenamiento
        elif image in val_images:
            shutil.copy(image_path, val_folder)    # Copiar a la carpeta de validación
        else:
            shutil.copy(image_path, test_folder)   # Copiar a la carpeta de prueba

    print(f"Imágenes divididas correctamente: {num_train} en train, {num_val} en val, {num_test} en test.")

def obtener_x_minimo(contorno):
    x, y, w, h = cv2.boundingRect(contorno)
    return x
def obtener_y_minimo(contorno):
    x, y, w, h = cv2.boundingRect(contorno)
    return y


def ungroup_pic(input_contours, output_path, info_file, axis="X"):
    n=1
    id_list=[]
    for pic in input_contours:
        try:
            print(f"Picture ungrouped {n}/{len(input_contours)}")
            pic_sin_ext = os.path.splitext(os.path.basename(pic.path))[0]
            list_contours_ordered=[]
            for contour in pic.masks.xy:
                array_contour=np.array(contour)
                array_contour=array_contour.reshape(-1,2)
                contour_pixels = array_contour.astype(np.int32)
                contour_opencv = contour_pixels.reshape((-1, 1, 2))
                list_contours_ordered.append(contour_opencv)
            if axis=="X":
                list_contours_ordered = sorted(list_contours_ordered, key=obtener_x_minimo)
            if axis=="Y":
                list_contours_ordered = sorted(list_contours_ordered, key=obtener_y_minimo)

            i=1
            for contour_ord in list_contours_ordered:
                imagen = cv2.imread(pic.path)
                mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)  # Crear una máscara negra
                cv2.drawContours(mascara, [contour_ord], -1, 255, -1)
                # Aplicar la máscara sobre la imagen para extraer la región
                region = cv2.bitwise_and(imagen, imagen, mask=mascara)
                # Encontrar el bounding box del contorno (la caja envolvente)
                x, y, w, h = cv2.boundingRect(contour_ord)
                # Recortar la región de interés (ROI) basada en el bounding box
                recorte_region = region[y:y+h, x:x+w]
                recorte_mascara = mascara[y:y+h, x:x+w]

                # Crear una imagen con un canal alfa (transparente fuera del contorno)
                imagen_transparente = np.zeros((h, w, 4), dtype=np.uint8)
                imagen_transparente[:, :, 0:3] = recorte_region  # Copiar la imagen recortada
                imagen_transparente[:, :, 3] = recorte_mascara   # Usar la máscara recortada como el canal alfa
                # Guardar la imagen como PNG con las dimensiones ajustadas al contorno
                output_folder=os.path.join(output_path,"Ungrouped_pics")
                os.makedirs(output_folder, exist_ok=True)
                name_pic=f'{output_folder}/{pic_sin_ext}_{i}.png'
                cv2.imwrite(name_pic, imagen_transparente)
                id_list.append([os.path.basename(pic.path),i,f"{pic_sin_ext}_{i}.png"])
                i=i+1
        except:
            print(f"Problem with the picture {pic_sin_ext}")
        n=n+1
    df_ungrouped = pd.DataFrame(id_list, columns=['Name_picture', 'Sample_number', 'Sample_picture'])
    info_data_completed=pd.merge(info_file,df_ungrouped,on=['Name_picture',"Sample_number"])
    output=os.path.join(output_path,"info_data_completed_ungrouped.txt")
    info_data_completed.to_csv(output, index=False, sep='\t')
    return info_data_completed

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

