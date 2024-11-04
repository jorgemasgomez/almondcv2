
import torch
import cv2
import numpy as np
from model_class import model_segmentation
import os
from aux_functions import midpoint, pol2cart, cart2pol
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
import pandas as pd

class pictures():
    def __init__(self, working_directory, input_folder,info_file,fruit, project_name,binary_masks=False):
        self.working_directory=working_directory
        self.input_folder=input_folder
        self.info_file=info_file
        self.fruit=fruit
        self.binary_masks=binary_masks

        self.project_name=project_name
        self.results_directory=os.path.join(working_directory,f"Results_{self.project_name}")
        os.makedirs(self.results_directory, exist_ok=True)

        self.path_export_1=os.path.join(self.results_directory, "results_morphology.txt")
        self.path_export_2=os.path.join(self.results_directory, "results_general.txt")
        self.path_export_3=os.path.join(self.results_directory, "pic_results")
        self.path_export_4=os.path.join(self.results_directory, "binary_masks")
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
        morphology_table = pd.DataFrame(columns=["Project_name","Sample_picture","Fruit_name", "Fruit_number","Length","Width","Area", "Perimeter","Hull_area","Solidity","Circularity","Ellipse_Ratio","L","a","b"])
        morphology_table.to_csv(self.path_export_1, mode='a', header=True, index=False, sep='\t')
        
        general_table=pd.DataFrame()
        general_table = pd.DataFrame(columns=["Name_picture","Sample_number","ID","Weight","Session","Shell","Pixelmetric","Sample_picture","N_fruits","Weight_per_fruit"])
        general_table.to_csv(self.path_export_2, mode='a', header=True, index=False, sep='\t')
        
        self.margin=margin

        for picture in self.segmentation_input:

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
            num_columns = int((len(mask_contours_list) + num_rows - 1) // num_rows)
            width = num_columns * spacing_x + 2 * margin
            image_base= np.zeros((height_pic, width, 3), dtype=np.uint8)
            # Generar centros de la cuadrícula
            x_coords = [margin + i * spacing_x for i in range(num_columns)]
            y_coords = [margin + i * spacing_y for i in range(num_rows)]
            centers = [(x, y) for y in y_coords for x in x_coords]

            
            measure_pic = image_base.copy()
            elipse_pic=image_base.copy()
            circ_pic=image_base.copy()

            #Lists
            all_almond_contours=[]
            count=1
            for contour in self.mask_contours_list:
                


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
                
                dimA = float(dA / pixelmetric)
                dimB = float(dB / pixelmetric)

                if dimA < dimB:
                    dimB_temp= dimB
                    dimB=dimA
                    dimA=dimB_temp
                else:
                    pass
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
                
                h_25=int(dimA*0.25)
                h_50=int(dimA*0.50)
                h_75=int(dimA*0.75)

                width_25=cv2.countNonZero(ROI[h_25:h_25 + 1, :])/pixelmetric
                width_50=cv2.countNonZero(ROI[h_50:h_50 + 1, :])/pixelmetric
                width_75=cv2.countNonZero(ROI[h_75:h_75 + 1, :])/pixelmetric

                print(width_25, width_50, width_75)
                
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
                        mascara = np.ones(image_base.shape[:2], dtype=np.uint8) * 255  # Máscara blanca

                        # Dibujar el contorno en la máscara, rellenar el interior de negro
                        cv2.drawContours(mascara, [cnt_rotated], -1, 0, thickness=cv2.FILLED)  # Rellenar con 0 (negro)

                        # Recortar la región de interés (ROI) de la imagen usando la bounding box
                        ROI = mascara[yb:yb+h, xb:xb+w]                   
                        
                        #Escalar el ancho a 1000

                        factor_escala=1000/ROI.shape[1]
                        nueva_h=int(ROI.shape[0]*factor_escala)
                        # Redimensionar la ROI para que tenga un alto de 1000 y el ancho calculado
                        ROI_redimensionada = cv2.resize(ROI,(1000, nueva_h))
                        # ROI_redimensionada=cv2.rotate(ROI_redimensionada, cv2.ROTATE_90_CLOCKWISE)
                        
                        # ROI_reducida = cv2.resize(ROI_redimensionada, (ROI_redimensionada.shape[1] // 2, ROI_redimensionada.shape[0] // 2))
                        # # cv2.imshow("ffs", ROI_reducida)
                        # # cv2.waitKey(0)
                        # # cv2.destroyAllWindows()

                        cv2.imwrite(f'{self.path_export_4}/{name_pic}_{count}.png', ROI_redimensionada)


                #Write_results
                row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,dimA, dimB, area[0], perimeter[0],area_hull[0], solidity[0], circularity, ellipse_rat, ml, ma, mb]],
                                  columns=morphology_table.columns)
                
                morphology_table = pd.concat([morphology_table, row], ignore_index=True)

                
                
                
                
                
                count=count+1

            row_general = self.info_file.loc[self.info_file['Sample_picture'] == name_pic]
            row_general=row_general.values.flatten().tolist()
            row_general.append(count-1)

            weight = self.info_file.loc[self.info_file['Sample_picture'] == name_pic, 'Weight'].values
            row_general.append(weight[0]/count-1)
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


            output_pic=np.concatenate((almonds_maskeadas, measure_pic, circ_pic, elipse_pic), axis=1)
            cv2.imwrite(f"{self.path_export_3}/rs_{name_pic}", output_pic)

        morphology_table.to_csv(self.path_export_1, mode='a', header=False, index=False, sep='\t')
        general_table.to_csv(self.path_export_2, mode='a', header=False, index=False, sep='\t')
         
          
            
            










