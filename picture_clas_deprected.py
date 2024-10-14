
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
    def __init__(self, working_directory, input_folder,info_file,fruit, model_seg, project_name, binary_masks=False,confidence_treshold=0.5, model_type='yolov8',
                slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS", check_result=False
                 , postprocess_match_metric="IOS", postprocess_match_threshold=0.5):
        self.working_directory=working_directory
        self.input_folder=input_folder
        self.info_file=info_file
        self.fruit=fruit
        self.model_seg=model_seg
        self.binary_masks=binary_masks
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device=device
        self.confidence_treshold=confidence_treshold
        self.model_type=model_type
        self.slice_height=slice_height
        self.slice_width=slice_width
        self.overlap_height_ratio=overlap_height_ratio
        self.overlap_width_ratio=overlap_width_ratio
        self.postprocess_type=postprocess_type
        self.postprocess_match_metric=postprocess_match_metric
        self.postprocess_match_threshold=postprocess_match_threshold
        self.check_result=check_result
        self.project_name=project_name
        self.results_directory=os.path.join(working_directory,f"Results_{self.project_name}")
        os.makedirs(self.results_directory, exist_ok=True)

        self.path_export_1=os.path.join(self.results_directory, "results_morphology.txt")
        self.path_export_2=os.path.join(self.results_directory, "results_general.txt")
        self.path_export_3=os.path.join(self.results_directory, "pic_results")
        self.path_export_4=os.path.join(self.results_directory, "binary_masks")
        os.makedirs(self.path_export_3, exist_ok=True)
        os.makedirs(self.path_export_4, exist_ok=True)


        if torch.cuda.is_available():
            # Obtener el índice del dispositivo actual
            gpu_index = torch.cuda.current_device()

            # Obtener el nombre de la GPU
            gpu_name = torch.cuda.get_device_name(gpu_index)

            # Obtener la memoria total de la GPU
            gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory

            # Convertir la memoria de bytes a gigabytes
            gpu_memory_gb = gpu_memory / (1024 ** 3)

            # Imprimir las características de la GPU
            print(f"GPU detectada: {gpu_name}")
            print(f"Memoria total de la GPU: {gpu_memory_gb:.2f} GB")
        else:
            print("No se detectó ninguna GPU. Usando CPU.")

    def segment(self):
        fruit_model=model_segmentation(working_directory=self.working_directory)
        contours_fruit=fruit_model.predict_model_sahi(model_path=self.model_seg, folder_input=self.input_folder,
                                                      confidence_treshold=self.confidence_treshold, model_type=self.model_type,
                                                      slice_height=self.slice_height, slice_width=self.slice_width,
                                                      overlap_height_ratio=self.overlap_height_ratio, overlap_width_ratio=self.overlap_width_ratio,
                                                      postprocess_type=self.postprocess_type, check_result=self.check_result,
                                                      postprocess_match_metric=self.postprocess_match_metric, postprocess_match_threshold=self.postprocess_match_threshold)
        self.contours_fruit=contours_fruit
        return self.contours_fruit
    
    def measure_almonds(self):
        morphology_table=pd.DataFrame()
        morphology_table = pd.DataFrame(columns=["Project_name","Sample_picture","Fruit_name", "Fruit_number","Length","Width","Area", "Perimeter","Hull_area","Solidity","Circularity","Ellipse_Ratio","L","a","b"])
        morphology_table.to_csv(self.path_export_1, mode='a', header=True, index=False, sep='\t')
        
        general_table=pd.DataFrame()
        general_table = pd.DataFrame(columns=["Name_picture","Sample_number","ID","Weight","Session","Shell","Pixelmetric","Sample_picture","N_fruits","Weight_per_fruit"])
        general_table.to_csv(self.path_export_2, mode='a', header=True, index=False, sep='\t')
        
        
        for contours in self.contours_fruit:
            name_pic=os.path.basename(contours[1])
            mask_contours_list=contours[0].object_prediction_list
            # Buscar la fila que contiene el valor y devolver el valor de 'Pixelmetric'
            pixelmetric = self.info_file.loc[self.info_file['Sample_picture'] == name_pic, 'Pixelmetric'].values
            
            almonds=cv2.imread(contours[1])






            #Pics to edit
            measure_pic = almonds.copy()
            elipse_pic=almonds.copy()
            circ_pic=almonds.copy()

            #Lists
            all_almond_contours=[]
            count=1
            for contour in mask_contours_list:
                contour=contour.mask.segmentation[0]
                array_contour=np.array(contour)
                array_contour=array_contour.reshape(-1,2)
                contour_pixels = array_contour.astype(np.int32)
                contour_opencv = contour_pixels.reshape((-1, 1, 2))
                #prueba reescalado
                scale_factor=0.5
                contour_opencv=(contour_opencv * scale_factor).astype(np.int32)
                #Refinado del contorno

                # h, w = almonds.shape[:2]
                # # Crear una máscara binaria a partir del contorno
                # mask = np.zeros((h, w), np.uint8)
                # # Dibujar el contorno en la máscara
                # cv2.drawContours(mask, [contour_opencv], -1, (255), thickness=cv2.FILLED)
                # # Crear un kernel para las operaciones morfológicas

                # mask_red= cv2.resize(mask, (w // 2, h // 2))
                # cv2.imshow('Contorno Refinado', mask_red)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # # Apply Gaussian blur
                # mask_refinada = cv2.GaussianBlur(mask, (5, 5), 0)
                # _, mask_refinada = cv2.threshold(mask_refinada, 127, 255, cv2.THRESH_BINARY)

                # # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

                # # # Aplicar operación de apertura para refinar los bordes del contorno
                # # mask_refinada=cv2.erode(mask, rect_kernel, iterations=2)
                # # mask_refinada=cv2.dilate(mask_refinada, rect_kernel, iterations=2)
                
                
                # mask_refinada_Red = cv2.resize(mask_refinada, (w // 2, h // 2))
                # cv2.imshow('Contorno Refinado', mask_refinada_Red)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # # Obtener nuevamente los contornos a partir de la máscara refinada
                # contour_opencv, _ = cv2.findContours(mask_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                # contour_opencv=contour_opencv[0]






                if cv2.contourArea(contour_opencv) < 600:
                    continue

                #las funciones fallan si hay un contorno con menos de 5 puntos, suele pasar cuando es pequeño o raro
                if  len(contour_opencv) < 5:
                    continue

                ############### Length and width ##########################
                (x,y),(MA,ma),angle = cv2.fitEllipse(contour_opencv)
                rotation_angle= 90-angle
                M = cv2.moments(contour_opencv)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cnt_norm = contour_opencv-[cx, cy]

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
                measure_pic=cv2.drawContours(measure_pic, [cnt_rotated], 0, (0, 255, 0), 1)

                box= cv2.boundingRect(cnt_rotated)
                (xb,yb, w, h)= box

                measure_pic=cv2.rectangle(measure_pic,(xb,yb),(xb+w,yb+h),(0,0,255),1)

                box= ((xb+(w/2),yb+(h/2)),(w,h), angle-90)
                

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

                
                ############### Area and perimeter##########################
                area=(cv2.contourArea(contour_opencv))/(pixelmetric**2)
                perimeter= (cv2.arcLength(contour_opencv, True))/(pixelmetric)
                ############### Hull Area and Solidity##########################

                hull = cv2.convexHull(contour_opencv)
                area_hull= (cv2.contourArea(hull))/(pixelmetric**2)
                solidity = area/area_hull
                ############### Circulatity and ellipse ##########################

                circularity=4*np.pi*(area/perimeter**2)
                circularity = float(circularity)
                (x,y),radius = cv2.minEnclosingCircle(contour_opencv)
                center = (int(x),int(y))
                radius = int(radius)
                circ_pic=cv2.circle(circ_pic,center,radius,(0,255,0),2)
                cv2.putText(circ_pic, "Circ: {:.2f}".format(circularity), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                ellipse = cv2.fitEllipse(contour_opencv)
                ellipse_rat= ellipse[1][0]/ellipse[1][1]
                elipse_pic= cv2.ellipse(elipse_pic, ellipse, (0,255,0),2)
                cv2.putText(elipse_pic, "Elip: {:.2f}".format(ellipse_rat), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                

                ############### Color Average ##########################


                lab_pic= cv2.cvtColor(almonds, cv2.COLOR_BGR2LAB)
                mask_contour = np.zeros_like(lab_pic)
                mask_contour=cv2.drawContours(mask_contour, [contour_opencv], -1, 255, -1)
                mask_contour= cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)

                color_pic=cv2.bitwise_and(lab_pic, lab_pic, mask=mask_contour)
                l, a, b= cv2.split(color_pic)
                
                ml=(np.mean(l[mask_contour != 0]))*100/255
                ma=(np.mean(a[mask_contour != 0]))-128
                mb=(np.mean(b[mask_contour != 0]))-128

                ############### Contours_mask ##########################
                #Esto es para la imagen final unicamente
                all_almond_contours.append(contour_opencv)
                #Aqui para exportar los contornos
                if self.binary_masks==True:
                    # Obtener la bounding box del contorno
                        x, y, w, h = cv2.boundingRect(cnt_rotated)

                        # Crear una máscara del tamaño de la imagen original, inicialmente blanca
                        mascara = np.ones(almonds.shape[:2], dtype=np.uint8) * 255  # Máscara blanca

                        # Dibujar el contorno en la máscara, rellenar el interior de negro
                        cv2.drawContours(mascara, [cnt_rotated], -1, 0, thickness=cv2.FILLED)  # Rellenar con 0 (negro)

                        # Recortar la región de interés (ROI) de la imagen usando la bounding box
                        ROI = mascara[y:y+h, x:x+w]
                        
                        #Escalar el ancho a 1000

                        factor_escala=1000/ROI.shape[1]
                        nueva_h=int(ROI.shape[0]*factor_escala)
                        # Redimensionar la ROI para que tenga un alto de 1000 y el ancho calculado
                        ROI_redimensionada = cv2.resize(ROI,(1000, nueva_h))
                        
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
            

            height_pic, width_pic = almonds.shape[:2]
            almonds_mask= np.zeros((height_pic, width_pic ), dtype=np.uint8)
            cv2.drawContours(almonds_mask, all_almond_contours, -1, 255, thickness=cv2.FILLED)
            almonds_maskeadas=cv2.bitwise_and(almonds, almonds, mask=almonds_mask)
            output_pic=np.concatenate((almonds_maskeadas, measure_pic, circ_pic, elipse_pic), axis=1)
            cv2.imwrite(f"{self.path_export_3}/rs_{name_pic}", output_pic)

        morphology_table.to_csv(self.path_export_1, mode='a', header=False, index=False, sep='\t')
        general_table.to_csv(self.path_export_2, mode='a', header=False, index=False, sep='\t')
         
          
            
            










