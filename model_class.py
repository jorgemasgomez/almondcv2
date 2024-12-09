import zipfile
import os
import shutil
import ultralytics
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import yaml
from sahi.predict import get_sliced_prediction,  AutoDetectionModel
from sahi.slicing import slice_image
from PIL import Image

class model_segmentation():
    def __init__(self, working_directory):
        self.working_directory=working_directory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device=device
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

    def train_segmentation_model(self, input_zip, pre_model="yolov8n-seg.pt", epochs=100, imgsz=640,batch=-1, name_segmentation="",
                                 retina_masks=True, pose=False, keypoints_pose=1):

        #Partimos del archivo en formato de YOLO Segmentation obtenido en CVAT

        input_zip_no_extension, extension=os.path.splitext(input_zip)
        output_folder_zip=os.path.join(self.working_directory,input_zip_no_extension)
        self.output_folder_zip=output_folder_zip
            # Verificar si la carpeta ya existe
        if os.path.exists(self.output_folder_zip):
        # Si existe, eliminar todo el contenido
            shutil.rmtree(self.output_folder_zip)
        os.makedirs(self.output_folder_zip, exist_ok=True)
        zip_file_path=os.path.join(self.working_directory,input_zip)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_folder_zip)

        # Modificamos el arccomohivo YAML para adaptarlo al input de YOLO training

        yaml_file = os.path.join(self.output_folder_zip,"data.yaml") 
        self.yaml_file=yaml_file

        with open(self.yaml_file, 'r') as file:  
            data = yaml.safe_load(file)

        # Crear un nuevo diccionario para los datos modificados
        modified_data = {
            'path': self.output_folder_zip,  # Cambiar 'path' a la primera fila
            'train': 'images/Train',  # Borrar referencia a Train.txt
            'val': 'images/Validation',  # Borrar referencia a Validation.txt
            'test': 'images/Test'  # Borrar referencia a Test.txt
        }
        if pose is True:
            modified_data = {
            'path': self.output_folder_zip,  # Cambiar 'path' a la primera fila
            'train': 'images/Train',  # Borrar referencia a Train.txt
            'val': 'images/Validation',  # Borrar referencia a Validation.txt
            'test': 'images/Test',
              "kpt_shape":[keypoints_pose, 3]    # Borrar referencia a Test.txt
        }
        
        # Agregar 'names' al final
        if 'names' in data:
            modified_data['names'] = data['names']
        # Escribir de nuevo el archivo YAML
        with open(self.yaml_file, 'w') as file:
            yaml.dump(modified_data, file, default_flow_style=False)


        #Creamos un directorio para guardar los resultados del training y de las predicciones

        results_models_directory=os.path.join(self.working_directory,f"results_models_segmentation_{name_segmentation}")
        self.results_models_directory=results_models_directory
        os.makedirs(self.results_models_directory, exist_ok=True)

        # Entrenamos el modelo
        model = YOLO(pre_model)  # load a pretrained model (recommended for training)
        model.to(self.device)
        model.train(data=self.yaml_file, epochs=epochs, imgsz=imgsz,batch=batch, project=self.results_models_directory, name="results_training")

        # Se habran generado los resultados para en los validación y training sets, para tener el del test set lo hacemos a continuación
        test_set_folder=os.path.join(output_folder_zip,"images/Test/")
        self.test_set_folder=test_set_folder
        
        #results devuelve una lista por foto, con una lista de arrays con las coordenadas de cada uno de los objetos segmentados.
        results_test_set = model.predict(self.test_set_folder, imgsz=imgsz, show= False, save=True, show_boxes=False, project=results_models_directory, save_txt=True,
                                          name="predictions_test", retina_masks=retina_masks)
        
    def predict_model(self, model_path, folder_input, imgsz=640, check_result=False, conf=0.6, max_det=300, retina_mask=True):
        model = YOLO(model_path)
        if check_result==False:
            results= model.predict(folder_input, imgsz=imgsz, show= False, save=False, show_boxes=False, conf=conf, max_det=max_det, retina_masks=retina_mask)
        elif check_result==True:
            results= model.predict(folder_input, imgsz=imgsz, show= False, save=True, show_boxes=False,project=self.working_directory, name="check_results", conf=conf, max_det=max_det, retina_masks=retina_mask)

        #results contiene una lista con los contornos de las imagenes, identificación y demas en results[i].path tienes la ruta, y en results[i].masks.xy[e] tienes el array de contorno de la foto i el contorno e
        
        return results
    
    #deprecated
    def predict_model_sahi(self, model_path, folder_input,confidence_treshold=0.5, model_type='yolov8',
                            slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS", check_result=False
                            , postprocess_match_metric="IOS", postprocess_match_threshold=0.5, retina_masks=True, imgsz=640):
        
        detection_model_seg = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_treshold,
        device=self.device,
        retina_masks=retina_masks,
        image_size=imgsz)
        
        # Lista de extensiones de imágenes comúnmente utilizadas
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        # Obtiene las rutas absolutas solo de archivos de imagen
        image_list = [os.path.join(folder_input, file) 
                    for file in os.listdir(folder_input) 
                    if os.path.splitext(file)[1].lower() in image_extensions]

        results_list=[]
        i=1
        for pic in image_list:
            print(f"Pic {i}/{len(image_list)}")

            try:
                result=get_sliced_prediction(
                    image=pic , detection_model=detection_model_seg, slice_height=slice_height,
                slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio,
                postprocess_type=postprocess_type, postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold, perform_standard_pred=True)
            except Exception as e:
                print(f"Error processing segmentation image {pic}: {e}")
                continue
            
            torch.cuda.empty_cache()
            # https://github.com/obss/sahi/blob/main/sahi/predict.py se pueden utilizar varios preprocesados, por si hay que probar
            results_list.append([result, pic])
            i=i+1
            if check_result==True:
                pic_sin_ext = os.path.splitext(os.path.basename(pic))[0]
                check_result_path=os.path.join(self.working_directory, "check_results")
                os.makedirs(check_result_path, exist_ok=True)
                result.export_visuals(export_dir=check_result_path, hide_labels=True, rect_th=1, file_name=f"prediction_result_{pic_sin_ext}")
            # os.remove(new_file_path_resized)
        return results_list

    
    
    def slice_predict_reconstruct(self, input_folder, imgsz, model_path, slice_width, slice_height, overlap_height_ratio, overlap_width_ratio, conf=0.5,retina_mask=True):
                # Lista de extensiones de imágenes comúnmente utilizadas
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        # Obtiene las rutas absolutas solo de archivos de imagen
        image_list = [os.path.join(input_folder, file) 
                    for file in os.listdir(input_folder) 
                    if os.path.splitext(file)[1].lower() in image_extensions]
        mask_list_images=[]
        n=1
        for image_path in image_list:
            print(f"Image{n}/{len(image_list)}")
            image_selected = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_selected.shape[2] == 4:
                image_selected = cv2.cvtColor(image_selected, cv2.COLOR_RGBA2RGB)
            
            image_sliced=slice_image(image=image_selected, slice_width=slice_width,
                                      slice_height=slice_height,overlap_height_ratio=overlap_height_ratio,
                                        overlap_width_ratio=overlap_width_ratio, verbose=True)
            
            

            slice_count=0
            mask_complete=np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)
            for slice in image_sliced.images:
                model = YOLO(model_path, verbose=False)  # load a pretrained model (recommended for training)
                model.to(self.device)
                results= model.predict(slice, imgsz=imgsz, show= False, save=False, show_boxes=False,
                                        verbose=False, conf=conf, retina_masks=retina_mask)
                
                h_slice=slice.shape[0]
                w_slice=slice.shape[1]

                mask_combined_slice = np.zeros((h_slice, w_slice), dtype=np.uint8)
                for result in results:
                    mask_combined_slice = np.zeros((h_slice, w_slice), dtype=np.uint8)
                    if result is None or result.masks is None or result.masks.data is None:
                        # Si no hay máscaras, crea una máscara negra del tamaño adecuado
                        continue  # Salta a la siguiente iteración
                    for j, mask in enumerate(result.masks.data):
                        mask = mask.cpu().numpy() * 255
                        # con cpu mask = mask.numpy() * 255
                        mask = cv2.resize(mask, (w_slice, h_slice))
                        mask_combined_slice = cv2.bitwise_or(mask_combined_slice, mask.astype(np.uint8))

                mask_added=np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)
                start_x=image_sliced.starting_pixels[slice_count][0]
                start_y=image_sliced.starting_pixels[slice_count][1]
                mask_added[start_y:start_y + h_slice, start_x:start_x + w_slice] = mask_combined_slice
                mask_complete = cv2.bitwise_or(mask_complete, mask_added)
                slice_count=slice_count+1
            n=n+1
            
            mask_list_images.append([mask_complete,image_path])
        return mask_list_images
    
            
            # # Crear una imagen verde (mismo tamaño que la original)
            # green_color = np.zeros_like(image_selected)
            # green_color[:] = [0, 255, 0]  # Color verde en formato BGR

            # # Crear una imagen con transparencia (40%)
            # alpha = 0.05  # Opacidad del 40%

            # # Crear la imagen final combinando la original y la verde
            # # Asegúrate de aplicar la máscara solo en las áreas que correspondan
            # # 1. Multiplica la máscara por el color verde
            # colored_mask = cv2.bitwise_and(green_color, green_color, mask=mask_complete )
            # # 2. Multiplica la imagen original por (1 - alpha)
            # #    Esto mantiene la parte de la imagen original en el área de la máscara.
            # original_weighted = cv2.multiply(image_selected, 1 - alpha)

            # # 3. Combina las dos imágenes
            # final_image = cv2.add(original_weighted, colored_mask)

            # # Mostrar la imagen final
            # cv2.imshow('Final Image', final_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
