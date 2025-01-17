
import torch
import cv2
import numpy as np
from model_class import model_segmentation
import os
from aux_functions import midpoint, pol2cart, cart2pol, calcular_simetria_horizontal, calcular_simetria_vertical
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import traceback
class pictures():
    def __init__(self, working_directory, input_folder,info_file,fruit, project_name,
                 binary_masks=False, blurring_binary_masks=False, blur_binary_masks_value=5, binary_pixel_size=250):
        self.working_directory=working_directory
        self.input_folder=input_folder
        self.info_file=info_file
        self.fruit=fruit
        self.binary_masks=binary_masks
        self.blurring_binary_masks=blurring_binary_masks
        self.blur_binary_masks_value=blur_binary_masks_value
        self.binary_pixel_size=binary_pixel_size


        self.project_name=project_name
        self.results_directory=os.path.join(working_directory,f"Results_{self.project_name}")
        os.makedirs(self.results_directory, exist_ok=True)

        self.path_export_1=os.path.join(self.results_directory, "results_morphology.txt")
        self.path_export_2=os.path.join(self.results_directory, "results_general.txt")
        self.path_export_3=os.path.join(self.results_directory, "pic_results")
        self.path_export_4=os.path.join(self.results_directory, "binary_masks")
        self.path_export_5=os.path.join(self.results_directory,"binary_masks_info_table.txt")
        self.path_export_outlier=os.path.join(self.results_directory,"outlier_table.txt")
        self.path_export_error=os.path.join(self.results_directory,"errors_table.txt")
        self.path_export_outlier_folder=os.path.join(self.results_directory, "outliers_pics")
        os.makedirs(self.path_export_3, exist_ok=True)
        os.makedirs(self.path_export_4, exist_ok=True)

    def set_postsegmentation_parameters(self,  segmentation_input, sahi=True, smoothing=False, kernel_smoothing=5, smoothing_iterations=2, watershed=False, kernel_watershed=5, threshold_watershed=0.7 ):
        
        self.sahi=sahi
        self.segmentation_input=segmentation_input
        if self.sahi==True:
            pass
        else:
            self.smoothing=smoothing
            self.smoothing_kernel=kernel_smoothing
            self.smooting_iterations=smoothing_iterations
            self.watershed=watershed
            self.kernel_watershed=kernel_watershed
            self.threshold_watershed=threshold_watershed

    
    def measure_almonds(self, margin=100, spacing=30):
        morphology_table=pd.DataFrame()
        morphology_table = pd.DataFrame(columns=["Project_name","Sample_picture","Fruit_name", "Fruit_number",
                                                 "Length","Width","Width_25","Width_50","Width_75","Area", "Perimeter","Hull_area",
                                                 "Solidity","Circularity","Ellipse_Ratio","L","a","b","Symmetry_v", "Symmetry_h","Shoulder_symmetry"])
        
        
        general_table=pd.DataFrame()
        general_table = pd.DataFrame(columns=["Name_picture","Sample_number","ID","Weight","Session","Shell","Pixelmetric","Sample_picture","N_fruits"])
        
        
        self.margin=margin
        error_list=[]
        for picture in self.segmentation_input:
            try:
                #Postsegmentation processing
                if self.sahi==True:
                    name_pic=os.path.basename(picture[1])
                    mask_contours_list=picture[0].object_prediction_list
                    almonds=cv2.imread(picture[1])

                    #cambio de formato y se guardan los contornos, esto habria que ver como optimizarlo hacemos dos veces el mismo loop pero es que necesitamos
                    #info antes del loop
                    sahi_contours_list=[]
                    for contour in mask_contours_list:
                        contour=contour.mask.segmentation[0]
                        array_contour=np.array(contour)
                        array_contour=array_contour.reshape(-1,2)
                        contour_pixels = array_contour.astype(np.int32)
                        contour = contour_pixels.reshape((-1, 1, 2))
                        sahi_contours_list.append(contour)
                    self.mask_contours_list=sahi_contours_list
                    
                else:
                    name_pic=os.path.basename(picture[1])
                    mask=picture[0]
                    if self.smoothing==True:
                        # # # Aplicar operación de apertura para refinar los bordes del contorno

                        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.smoothing_kernel, self.smoothing_kernel))
                        mask=cv2.erode(mask, rect_kernel, iterations=self.smooting_iterations)
                        mask=cv2.dilate(mask, rect_kernel, iterations=self.smooting_iterations)

                    if self.watershed==True:
                        
                        # # # Aplicar operación de apertura para refinar los bordes del contorno y para quitar ruido
                        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.smoothing_kernel, self.smoothing_kernel))
                        mask=cv2.erode(mask, rect_kernel, iterations=self.smooting_iterations)
                        mask=cv2.dilate(mask, rect_kernel, iterations=self.smooting_iterations)

                        # Determinar el fondo seguro (dilatando)
                        sure_bg = cv2.dilate(mask, rect_kernel, iterations=3)  # Fondo seguro dilatado

                        # Usar la transformada de distancia para obtener el foreground (primer plano) seguro
                        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, self.kernel_watershed)
                        _, sure_fg = cv2.threshold(dist_transform, self.threshold_watershed * dist_transform.max(), 255, 0)

                        # Encontrar las áreas desconocidas (región que no es fondo ni foreground seguro)
                        sure_fg = np.uint8(sure_fg)
                        unknown = cv2.subtract(sure_bg, sure_fg)

                        # Marcar los objetos con etiquetas distintas
                        _, markers = cv2.connectedComponents(sure_fg)

                        # Aumentar los marcadores en 1, para que el fondo sea 1 y no 0
                        markers = markers + 1

                        # Marcar la región desconocida (áreas de borde) con 0
                        markers[unknown == 255] = 0

                        # Aplicar Watershed a la imagen en color (si la máscara es binaria, convertimos a color)
                        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convertir la máscara a color para usar Watershed
                        markers = cv2.watershed(img_color, markers)

                        # Marcar los bordes detectados por Watershed como -1 (normalmente son las líneas de división entre objetos)
                        mask[markers == -1] = 0  # Puedes visualizar los bordes o eliminarlos

                        # Convertir la imagen resultante de Watershed a una máscara binaria para encontrar contornos
                        mask = np.uint8(markers > 1) * 255  # Objetos correctamente segmentados
                        
                        self.mask_contours_list, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        almonds=cv2.imread(picture[1])
                        
                

                # Buscar la fila que contiene el valor y devolver el valor de 'Pixelmetric'
                pixelmetric = self.info_file.loc[self.info_file['Sample_picture'] == name_pic, 'Pixelmetric'].values
                #Pics to edit

                height_pic, width_pic = almonds.shape[:2]
                # almonds_mask_3= np.zeros((height_pic, width_pic,3 ), dtype=np.uint8)



                #Preparar cuadricula 
                # Calcular el tamaño máximo de los contornos
                max_height = max(cv2.boundingRect(cnt)[3] for cnt in self.mask_contours_list)
                max_width = max(cv2.boundingRect(cnt)[2] for cnt in self.mask_contours_list)

                # Definir espaciado basado en el tamaño máximo de los contornos
                spacing_x = max_width + spacing  # Añadimos un margen adicional para separación
                spacing_y = max_height + spacing
                
                num_rows = int((height_pic - 2 * self.margin) / spacing_y)
                num_columns = int((len(self.mask_contours_list) + num_rows - 1) // num_rows)
                width = num_columns * spacing_x + 2 * margin
                image_base= np.zeros((height_pic, width, 3), dtype=np.uint8)
                # Generar centros de la cuadrícula
                x_coords = [margin + i * spacing_x for i in range(num_columns)]
                y_coords = [margin + i * spacing_y for i in range(num_rows)]
                centers = [(x, y) for y in y_coords for x in x_coords]

                
                measure_pic = image_base.copy()
                elipse_pic=image_base.copy()
                circ_pic=image_base.copy()
                widths_pic=image_base.copy()
                # symmetry_pic=image_base.copy()
                #Lists
                all_almond_contours=[]
                count=1
                for contour in self.mask_contours_list:
                    try:


                        if cv2.contourArea(contour) < 600:
                            continue

                        #las funciones fallan si hay un contorno con menos de 5 puntos, suele pasar cuando es pequeño o raro
                        if  len(contour) < 5:
                            continue

                        ############### Length, width , dist_maxima ##########################
                        (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
                        rotation_angle= 180-angle
                        M = cv2.moments(contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cnt_norm = contour-[cx, cy]

                        coordinates = cnt_norm[:, 0, :]
                        xs, ys = coordinates[:, 0], coordinates[:, 1]
                        thetas, rhos = cart2pol(xs, ys)
                                                
                        thetas_deg = np.rad2deg(thetas)
                        thetas_new_deg = (thetas_deg + rotation_angle) % 360
                        thetas_new = np.deg2rad(thetas_new_deg)

                        xs, ys = pol2cart(thetas_new, rhos)
                        cnt_norm[:, 0, 0] = xs
                        cnt_norm[:, 0, 1] = ys
                        cnt_rotated = cnt_norm + [cx, cy]
                        cnt_rotated = cnt_rotated.astype(np.int32)
                        #calculamos desplazamiento 
                        # Calculamos el desplazamiento
                       

                        cx_cen, cy_cen = centers[count-1]
                        #calculamos desplazamientos para despues
                        dx = cx_cen - cx  # Desplazamiento en x
                        dy = cy_cen - cy  # Desplazamiento en y
                        #rotamos
                        cnt_rotated= cnt_rotated - [cx, cy] + [cx_cen, cy_cen]
                        cnt_rotated = cnt_rotated.astype(np.int32)
                        cx_initial,cy_initial=[cx, cy]
                        cx, cy=[cx_cen, cy_cen]

                        
                        #calcular punto mas lejano para rotar si estan al reves
                        dist_maxima = 0
                        punto_mas_lejano = None

                        for punto in cnt_rotated:
                            # Coordenadas del punto actual
                            x, y = punto[0]
                            # Calcular la distancia euclidiana entre el punto y el centroide
                            distancia = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            
                            # Si la distancia es la mayor encontrada hasta ahora, actualizamos
                            if distancia > dist_maxima:
                                dist_maxima = distancia
                                punto_mas_lejano = (x, y)
                        giro=0
                        if punto_mas_lejano[1] < cy:
                            # print(count, name_pic)
                            # Normalizar el contorno restando el centro de masa
                            contour_centered = cnt_rotated - [cx, cy]

                            # Rotar 180º multiplicando por -1
                            contour_rotated = contour_centered * [-1, -1]

                            # Volver a trasladar el contorno al centro original
                            contour_rotated = contour_rotated + [cx, cy]
                            cnt_rotated = contour_rotated.astype(np.int32)
                            giro=180
                        



                        
                        #Poner máscara de relleno
                        relleno=almonds.copy()
                        # Definir el ángulo de rotación y el desplazamiento
                        # Rotación en grados
                        # Desplazamiento en píxeles

                        # Obtener el centro de la imagen para rotar alrededor de este
                        # centro_x_relleno, centro_y_relleno = relleno.shape[1] // 2, relleno.shape[0] // 2   

                        # Crear la matriz de rotación
                        
                        M_rotacion = cv2.getRotationMatrix2D((cx_initial, cy_initial), giro-rotation_angle, 1.0)

                        # Agregar la traslación a la matriz de rotación
                        M_rotacion[0, 2] += dx  # Desplazar en x
                        M_rotacion[1, 2] += dy  # Desplazar en y

                        image_transformada = cv2.warpAffine(relleno, M_rotacion, (image_base.shape[1], image_base.shape[0]))
                        

                        # 2. Crear una máscara del mismo tamaño que la imagen transformada
                        mascara = np.zeros(image_transformada.shape[:2], dtype=np.uint8)
                        # 3. Dibujar el contorno en la máscara y rellenar el interior
                        mascara=cv2.drawContours(mascara, [cnt_rotated], -1, 255, thickness=cv2.FILLED)

                        relleno_almendra = cv2.bitwise_and(image_transformada, image_transformada , mask=mascara)
                        relleno_almendra = relleno_almendra[:, :image_base.shape[1], :]
                        
                        # extend_pic=image_base.copy()
                        # # Copiar relleno_almendra en la nueva imagen
                        # extend_pic[:, :relleno_almendra.shape[1]] = relleno_almendra 
                        # print(count)
                        # print(relleno_almendra.shape, "relleno")
                        # print(measure_pic.shape, "measure")
                        # print(extend_pic.shape,"extend")
                        # print("centro:", cy, cx )
                        # # cv2.imshow("dfdfs", relleno_almendra)
                        # # cv2.waitKey()
                        # # cv2.destroyAllWindows()

                        measure_pic = cv2.bitwise_or(measure_pic, relleno_almendra)
                        circ_pic = cv2.bitwise_or(circ_pic, relleno_almendra)
                        elipse_pic = cv2.bitwise_or(elipse_pic, relleno_almendra)
                        widths_pic=cv2.bitwise_or(widths_pic, relleno_almendra)
                        # symmetry_pic=cv2.bitwise_or(symmetry_pic, relleno_almendra)

                        measure_pic=cv2.drawContours(measure_pic, [cnt_rotated], 0, (0, 255, 0), 1)

                        box= cv2.boundingRect(cnt_rotated)
                        (xb,yb, w, h)= box

                        measure_pic=cv2.rectangle(measure_pic,(xb,yb),(xb+w,yb+h),(0,0,255),1)

                        box= ((xb+(w/2),yb+(h/2)),(w,h), angle-180)
                        

                        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box = np.array(box, dtype="int")
                        
                        box = perspective.order_points(box)
                        
                        (tl, tr, br, bl) = box
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)
                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)
                        
                        
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        
                        if dA < dB:
                            dB_temp= dB
                            dB=dA
                            dA=dB_temp
                        else:
                            pass

                        dimA = float(dA / pixelmetric)
                        dimB = float(dB / pixelmetric)
                        cv2.putText(measure_pic, f"N: {count}", (int(tltrX), int(tltrY+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.putText(measure_pic, "L:{:.2f}mm".format(dimA), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.putText(measure_pic, "W:{:.2f}mm".format(dimB), (int(trbrX - 100), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        
                        
                        
                        
                        
                        
                        
                        
                        ########## Width 25, 50 y 75 ############
                        # Crear una máscara del tamaño de la imagen original,
                        mascara = np.zeros(image_base.shape[:2], dtype=np.uint8)   

                        # Dibujar el contorno en la máscara, rellenar el interior 
                        cv2.drawContours(mascara, [cnt_rotated], -1, 255, thickness=cv2.FILLED)  

                        # Recortar la región de interés (ROI) de la imagen usando la bounding box
                        ROI = mascara[yb:yb+h, xb:xb+w]
                        
                        # print(h, "h")
                        # print(dA, "dA")
                        h_25=int(dA*0.25)
                        h_50=int(dA*0.50)
                        h_75=int(dA*0.75)

                        width_25=cv2.countNonZero(ROI[h_25:h_25 + 1, :])/pixelmetric
                        width_50=cv2.countNonZero(ROI[h_50:h_50 + 1, :])/pixelmetric
                        width_75=cv2.countNonZero(ROI[h_75:h_75 + 1, :])/pixelmetric
                        
                        verde = (0, 255, 0)
                        # Dibujar la línea verde en el 25% de la altura
                        cv2.line(widths_pic, (xb, yb + h_25), (xb + w, yb + h_25), verde, 1)

                        # Dibujar la línea verde en el 50% de la altura
                        cv2.line(widths_pic, (xb, yb + h_50), (xb + w, yb + h_50), verde, 1)

                        # Dibujar la línea verde en el 75% de la altura
                        cv2.line(widths_pic, (xb, yb + h_75), (xb + w, yb + h_75), verde, 1)

                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_25)), (int(tltrX - 40), int(yb + h_25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_50)), (int(tltrX - 40), int(yb + h_50)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_75)), (int(tltrX - 40), int(yb + h_75)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        # Shoulder
                        
                        # Suponiendo que tienes la máscara y las coordenadas (xb, yb, w, h) ya definidas
                        # Tomar la fila yb + int(h*0.25) de la máscara
                        row = mascara[yb + int(h * 0.25), :]  # Extraemos la fila completa de la máscara

                        # Encontrar el primer y último píxel con valor 255 en esa fila
                        start_col = np.argmax(row == 255)  # Primer píxel con valor 255
                        end_col = len(row) - np.argmax(row[::-1] == 255)  # Último píxel con valor 255
                        # Verificar que los índices sean válidos (que efectivamente haya píxeles con valor 255)
                        if start_col < end_col:

                            ROI_shoulder = mascara[yb:yb + int(h * 0.25), start_col:end_col]  # ROI entre los dos índices
                                                # Escalar la ROI al tamaño que desees (por ejemplo, duplicar el tamaño)
                            # scaled_ROI = cv2.resize(ROI_shoulder, (0, 0), fx=2, fy=2)  # Escala 2x en ambos ejes
                            
                            shoulder=calcular_simetria_vertical(ROI_shoulder)

                            # print(shoulder)
                            # cv2.putText(widths_pic, "{:.3f}".format(float(shoulder)), (int(tltrX - 120), int(yb + h_75)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                            # # Mostrar la imagen escalada
                            # cv2.imshow("Escalada ROI", scaled_ROI)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                        else:
                            ROI_shoulder = None  # No se encontró una región con valor 255 en esa filaROI_shoulder = mascara[yb:yb+int(h*0.25), xb:xb+w]
                            print("shoulder_error")



                        




                        
                        ############### Symmetry x and y ##########################
                        
                        simmetry_v=calcular_simetria_vertical(ROI)
                        simmetry_h=calcular_simetria_horizontal(ROI)

                        # cv2.line(symmetry_pic, (xb, yb + h_50), (xb + w, yb + h_50), verde, 1)
                        # cv2.line(symmetry_pic, (xb+int(w/2), yb), (xb + int(w/2), yb + h), verde, 1)

                        # cv2.putText(symmetry_pic, "Sv: {:.2f}".format(float(simmetry_v)), (int(tltrX - 40), int(yb )), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        # cv2.putText(symmetry_pic, "Sh: {:.2f}".format(float( simmetry_h)), (int(tltrX - 40), int(yb + h_50)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


                        ############### Area and perimeter##########################
                        area=(cv2.contourArea(cnt_rotated))/(pixelmetric**2)
                        perimeter= (cv2.arcLength(cnt_rotated, True))/(pixelmetric)
                        ############### Hull Area and Solidity##########################

                        hull = cv2.convexHull(cnt_rotated)
                        area_hull= (cv2.contourArea(hull))/(pixelmetric**2)
                        solidity = area/area_hull
                        ############### Circulatity and ellipse ##########################

                        circularity=4*np.pi*(area/perimeter**2)
                        circularity = float(circularity)
                        (x,y),radius = cv2.minEnclosingCircle(cnt_rotated)
                        center = (int(x),int(y))
                        radius = int(radius)
                        circ_pic=cv2.circle(circ_pic,center,radius,(0,255,0),2)
                        cv2.putText(circ_pic, "Circ: {:.2f}".format(circularity), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                        ellipse = cv2.fitEllipse(cnt_rotated)
                        ellipse_rat= ellipse[1][0]/ellipse[1][1]
                        elipse_pic= cv2.ellipse(elipse_pic, ellipse, (0,255,0),2)
                        cv2.putText(elipse_pic, "Elip: {:.2f}".format(ellipse_rat), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        

                        ############### Color Average ##########################


                        lab_pic= cv2.cvtColor(almonds, cv2.COLOR_BGR2LAB)
                        mask_contour = np.zeros_like(lab_pic)
                        mask_contour=cv2.drawContours(mask_contour, [contour], -1, 255, -1)
                        mask_contour= cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)

                        color_pic=cv2.bitwise_and(lab_pic, lab_pic, mask=mask_contour)
                        l, a, b= cv2.split(color_pic)
                        
                        ml=(np.mean(l[mask_contour != 0]))*100/255
                        ma=(np.mean(a[mask_contour != 0]))-128
                        mb=(np.mean(b[mask_contour != 0]))-128

                        ############### Contours_mask ##########################
                        #Esto es para la imagen final unicamente
                        all_almond_contours.append(contour)
                        #Aqui para exportar los contornos
                        if self.binary_masks==True:
                        # Obtener la bounding box del contorno    
                                # Obtener la bounding box del contorno    
                                # Crear la máscara base y el contorno
                                
                                # Crear la máscara base y el contorno
                                # Crear la máscara base y el contorno
                                mascara = np.ones(image_base.shape[:2], dtype=np.uint8) * 255  # Máscara blanca
                            
                                # cnt_rotated = cv2.approxPolyDP(cnt_rotated, epsilon=2, closed=True)

                                # Dibujar el contorno en la máscara, rellenar con negro
                                cv2.drawContours(mascara, [cnt_rotated], -1, 0, thickness=cv2.FILLED)  # Rellenar con 0 (negro)
                                


                                # Recortar la región de interés (ROI) de la imagen usando la bounding box
                                ROI = mascara[yb:yb+h, xb:xb+w]                   

                                factor_escala = self.binary_pixel_size / ROI.shape[0]
                                nueva_h = self.binary_pixel_size
                                nueva_w = int(ROI.shape[1] * factor_escala)

                                # Redimensionar usando interpolación nearest-neighbor
                                # ROI_redimensionada = cv2.resize(ROI, (nueva_w*2, nueva_h*2), interpolation=cv2.INTER_NEAREST)
                                ROI_redimensionada = cv2.resize(ROI, (nueva_w, nueva_h), interpolation=cv2.INTER_NEAREST_EXACT)

                                # Invertir la máscara para que la figura sea blanca (255) y el fondo negro (0)
                                ROI_invertida = cv2.bitwise_not(ROI_redimensionada)

                                # Crear kernel para erosión y dilatación
                            
                                # Suavizar los bordes: primero dilatación, luego erosión
                                # ROI_dilatada = cv2.dilate(ROI_invertida, rect_kernel, iterations=self.smooting_iterations)  # Expandir bordes de la figura
                                # ROI_suavizada = cv2.erode(ROI_dilatada, rect_kernel, iterations=self.smooting_iterations)   # Reducir bordes para suavizar

                                # Invertir la máscara de nuevo para restaurar la figura en negro y fondo en blanco
                                # ROI_final = cv2.bitwise_not(ROI_suavizada)
                                ROI_final = cv2.bitwise_not(ROI_invertida)

                                # Creamos una imagen de fondo blanco de 1000x1000 píxeles
                                imagen_final = np.ones((self.binary_pixel_size, self.binary_pixel_size), dtype=np.uint8) * 255

                                # Calculamos las coordenadas para centrar la ROI en la imagen final
                                x_offset = (self.binary_pixel_size - nueva_w) // 2
                                y_offset = (self.binary_pixel_size - nueva_h) // 2

                                # Insertamos la ROI suavizada en el fondo blanco
                                imagen_final[y_offset:y_offset + nueva_h, x_offset:x_offset + nueva_w] = ROI_final
                                
                                _, imagen_final_binaria = cv2.threshold(imagen_final, 254, 255, cv2.THRESH_BINARY)


                                # Encuentra las coordenadas (x, y) de todos los píxeles negros (valor 0)
                                coordenadas_negras = np.column_stack(np.where(imagen_final_binaria == 0))

                                if len(coordenadas_negras) > 0:
                                    # Encuentra el píxel negro con el mayor valor de Y (parte inferior máxima)
                                    max_y = np.max(coordenadas_negras[:, 0])  # Máximo valor en el eje Y

                                    # Filtra los píxeles que están en la fila más baja (eje Y = max_y)
                                    pixels_fila_mas_baja = coordenadas_negras[coordenadas_negras[:, 0] == max_y]

                                    # Encuentra el píxel más a la derecha (máximo X) entre los filtrados
                                    pixel_mas_derecha = pixels_fila_mas_baja[np.argmax(pixels_fila_mas_baja[:, 1])]

                                #     # print("Píxel negro más bajo en el eje Y (parte inferior máxima):", (pixel_mas_derecha[1], max_y))
                                #     # print("Píxel negro más a la derecha en esa fila:", pixel_mas_derecha)
                                # else:
                                #     print("No hay píxeles negros en la imagen.")

                                if self.blurring_binary_masks is True:
                                    # Aplicar suavizado gaussiano si está habilitado
                                    imagen_final = cv2.GaussianBlur(imagen_final, (self.blur_binary_masks_value, self.blur_binary_masks_value), 0)
                                    # Recupera un borde definido aplicando un umbral
                                    _, imagen_final  = cv2.threshold(imagen_final , 127, 255, cv2.THRESH_BINARY)
                                

                                if pixel_mas_derecha[1] > self.binary_pixel_size/2:
                                    # print("IMAGEN GIRADA", name_pic, count)
                                    imagen_final = cv2.flip(imagen_final, 1)

                                cv2.imwrite(f'{self.path_export_4}/{name_pic}_{count}.jpg', imagen_final)

                    except Exception as e:
                        print(f"Error with picture {name_pic}", f"Fruit number: {count}", e)
                        error_list.append([name_pic, count, e])
                        traceback.print_exc()
                        row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,None, None, None, None, None,None, None,None, None, None, None, None, None, None, None, None, None]],
                        columns=morphology_table.columns)
                        morphology_table = pd.concat([morphology_table, row], ignore_index=True)
                        count=count+1
                        
                                

                    #Write_results
                    row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,dimA, dimB, width_25[0], width_50[0], width_75[0],area[0], perimeter[0],area_hull[0], solidity[0], circularity, ellipse_rat, ml, ma, mb, simmetry_h, simmetry_v, shoulder]],
                                    columns=morphology_table.columns)
                    
                    morphology_table = pd.concat([morphology_table, row], ignore_index=True)

                    
                    
                    
                    
                    
                    count=count+1

                row_general = self.info_file.loc[self.info_file['Sample_picture'] == name_pic]
                row_general=row_general.values.flatten().tolist()
                row_general.append(count-1)

                row_general=pd.DataFrame([row_general], columns=general_table.columns)
                general_table=pd.concat([general_table,row_general], ignore_index=True)

                
                #Draw the contours for green masking
                almonds_mask= np.zeros((height_pic, width_pic), dtype=np.uint8)
                cv2.drawContours(almonds_mask, all_almond_contours, -1, 255, thickness=cv2.FILLED)
                



                #Este chunk es para que se exporte solo lo que es el contenido de la mascara
                # height_pic, width_pic = almonds.shape[:2]
                # almonds_mask= np.zeros((height_pic, width_pic ), dtype=np.uint8)
                # cv2.drawContours(almonds_mask, all_almond_contours, -1, 255, thickness=cv2.FILLED)
                # almonds_maskeadas=cv2.bitwise_and(almonds, almonds, mask=almonds_mask)

                #Este chunk pone una mascara transparente verde sobre las almendras

                # # Crear una imagen verde (mismo tamaño que la original)
                green_color = np.zeros_like(almonds)
                green_color[:] = [0, 255, 0]  # Color verde en formato BGR

                # # Crear una imagen con transparencia (40%)
                alpha = 0.05  # Opacidad del 40%

                
                # # Crear la imagen final combinando la original y la verde
                # # Asegúrate de aplicar la máscara solo en las áreas que correspondan
                # # 1. Multiplica la máscara por el color verde
                colored_mask = cv2.bitwise_and(green_color, green_color, mask=almonds_mask )
                # # 2. Multiplica la imagen original por (1 - alpha)
                # #    Esto mantiene la parte de la imagen original en el área de la máscara.
                original_weighted = cv2.multiply(almonds, 1 - alpha)

                # # 3. Combina las dos imágenes
                almonds_maskeadas= cv2.add(original_weighted, colored_mask)


                # output_pic=np.concatenate((almonds_maskeadas, measure_pic, widths_pic, circ_pic, elipse_pic, symmetry_pic), axis=1)
                output_pic=np.concatenate((almonds_maskeadas, measure_pic, widths_pic, circ_pic, elipse_pic), axis=1)
                cv2.imwrite(f"{self.path_export_3}/rs_{name_pic}", output_pic)
            except Exception as e:
                print(f"Error with picture {name_pic}", e)
                error_list.append([name_pic, "General", e])
                
        morphology_table = pd.merge(morphology_table, general_table[['Sample_picture', 'ID']], left_on='Sample_picture', right_on='Sample_picture', how='left')
        if self.binary_masks is True:
            binary_table = morphology_table[['Sample_picture', 'ID', 'Fruit_number']]
            binary_table['Binary_mask_picture'] = binary_table['Sample_picture'] + '_' + binary_table['Fruit_number'].astype(str)
            binary_table = binary_table[['Binary_mask_picture', 'ID']]
            binary_table.to_csv(self.path_export_5, mode='w', header=True, index=False, sep='\t')
            self.binary_table=binary_table

        morphology_table.to_csv(self.path_export_1, mode='w', header=True, index=False, sep='\t')
        general_table.to_csv(self.path_export_2, mode='w', header=True, index=False, sep='\t')

        self.morphology_table=morphology_table
        self.general_table=general_table

        #####OUTLIER DETECTION #############
        # Define las columnas que deseas en el DataFrame
        columns = ["Sample_picture", "Fruit_number", "Cause"]

        # Convierte la lista a un DataFrame
        error_df = pd.DataFrame(error_list, columns=columns)
        error_df.to_csv(self.path_export_error, mode='w', header=True, index=False, sep='\t')

        outlier_df=pd.DataFrame()
        outlier_df = pd.DataFrame(columns=["Sample_picture","Fruit_number", "Causes"])

        # Define las columnas a analizar
        columns_to_check = ["Width_25", "Width_75", "Area", "Circularity", "Symmetry_v", "Symmetry_h"]

        
        # Agrupa por Sample_picture y analiza cada grupo
        for sample_picture, group in self.morphology_table.groupby("Sample_picture"):
            # Itera por las columnas a analizar
            for col in columns_to_check:
                # Calcula la media y la desviación estándar
                mean = group[col].mean()
                std_dev = group[col].std()
                
                # Define los límites para identificar outliers
                lower_bound = mean - 3 * std_dev
                upper_bound = mean + 3 * std_dev

                # Filtra los outliers
                outliers = group[(group[col] < lower_bound) | (group[col] > upper_bound)]

                # Agrega las filas de outliers al DataFrame de resultados
                for idx, row in outliers.iterrows():
                    fruit_number = row["Fruit_number"]
                    # Si ya existe en outlier_df, agrega la nueva causa
                    if ((outlier_df["Sample_picture"] == sample_picture) & (outlier_df["Fruit_number"] == fruit_number)).any():
                        # Encuentra la fila correspondiente
                        current_index = outlier_df[(outlier_df["Sample_picture"] == sample_picture) & (outlier_df["Fruit_number"] == fruit_number)].index[0]
                        outlier_df.at[current_index, "Causes"] += f", {col}"
                    else:
                        # Crea una nueva entrada
                        outlier_df = pd.concat([
                            outlier_df,
                            pd.DataFrame({
                                "Sample_picture": [sample_picture],
                                "Fruit_number": [fruit_number],
                                "Causes": [col]
                            })
                        ], ignore_index=True)



        print(error_list)
        print(outlier_df)
        outlier_df.to_csv(self.path_export_outlier, mode='w', header=True, index=False, sep='\t')
        if not os.path.exists(self.path_export_outlier_folder):
            os.makedirs(self.path_export_outlier_folder)

        # Iterar sobre las filas de outlier_df
        for _, row in outlier_df.iterrows():
            sample_picture = row["Sample_picture"]
             # Añadir el prefijo 'rs_' al nombre del archivo
            sample_picture = f"rs_{sample_picture}"
            # Construir la ruta de la imagen original
            image_path = os.path.join(self.path_export_3, sample_picture)
            
            # Verificar si la imagen existe antes de copiarla
            if os.path.exists(image_path):
                # Construir la ruta de destino en la carpeta outliers_pictures
                destination = os.path.join(self.path_export_outlier_folder, sample_picture)
                
                # Copiar la imagen con el nuevo nombre
                shutil.copy(image_path, destination)
            else:
                print(f"Advertencia: La imagen {sample_picture} no se encuentra en {self.path_export_3}")
                
        
        #crear función corrige outliers en outputs general table,  binary masks y output image


    def correct_outliers(self, outlier_file=None, addingsamples_file=None):


        #Para añadir o quitar al numero de frutos utilizamos adding samples_file

        if addingsamples_file is not None:
            add_df = pd.read_csv(addingsamples_file, delimiter="\t")
            general_table_cleaned=self.general_table


            # Realizamos un merge (join) entre los DataFrames basado en la columna 'Name_picture'
            merged_df = general_table_cleaned.merge(add_df, on="Sample_picture", how="left")

            # Convertimos las columnas a enteros antes de la suma, si no lo son
            merged_df["N_fruits"] = merged_df["N_fruits"].fillna(0).astype(int)
            merged_df["n_almonds_to_Add"] = merged_df["n_almonds_to_Add"].fillna(0).astype(int)

            # Realizamos la suma
            merged_df["N_fruits"] = merged_df["N_fruits"] + merged_df["n_almonds_to_Add"]


            # Eliminamos la columna adicional si ya no la necesitamos
            merged_df = merged_df.drop(columns=["n_almonds_to_Add"])

            # Si necesitas actualizar tu DataFrame original:
            general_table_cleaned = merged_df
            general_table_cleaned.to_csv(self.path_export_2, mode='w', header=True, index=False, sep='\t')

        if outlier_file is not None:
            outlier_df_2 = pd.read_csv(outlier_file, delimiter="\t")

            #ELIMINAR DE MORPHOLOGY TABLE
            # Filtramos el DataFrame morphology_table para eliminar las filas que tengan combinaciones de Sample_picture y Fruit_number en outlier_df
            morphology_table_cleaned = self.morphology_table[~self.morphology_table[['Sample_picture', 'Fruit_number']].apply(tuple, 1).isin(outlier_df_2[['Sample_picture', 'Fruit_number']].apply(tuple, 1))]
            # Verificamos el resultado
            morphology_table_cleaned.to_csv(self.path_export_1, mode='w', header=True, index=False, sep='\t')
            
            #ELIMINAR DE LA TABLA DE BINARY
            # Crear una columna temporal en el DataFrame de outliers con la forma de los valores en Binary_mask_picture
            
            outlier_df_2['Binary_mask_picture'] = outlier_df_2['Sample_picture'] + "_" + outlier_df_2['Fruit_number'].astype(str)

            # Filtrar el DataFrame de binary masks para eliminar las filas que coincidan con los valores en outlier_df
            binary_table_cleaned = self.binary_table[~self.binary_table['Binary_mask_picture'].isin(outlier_df_2['Binary_mask_picture'])]
            binary_table_cleaned.to_csv(self.path_export_5, mode='w', header=True, index=False, sep='\t')
            
            #ELIMINAR LAS BINARY MASKS
            # Crear la lista de máscaras a eliminar a partir de outlier_df
            outlier_df_2['Mask_filename'] = outlier_df_2['Sample_picture'] + "_" + outlier_df_2['Fruit_number'].astype(str) + ".jpg"

            # Iterar sobre los archivos en la carpeta de máscaras
            for mask_filename in os.listdir(self.path_export_4):
                # Verificar si el archivo está en la lista de máscaras a eliminar
                if mask_filename in outlier_df_2['Mask_filename'].values:
                    # Construir la ruta completa del archivo
                    file_path = os.path.join(self.path_export_4, mask_filename)
                    # Eliminar el archivo
                    os.remove(file_path)
                    print(f"Archivo eliminado: {mask_filename}")
        print(" PLEASE CHECK NUMBER OF FRUITS IN GENERAL TABLE IF YOU ARE WANT WEIGHT.")


