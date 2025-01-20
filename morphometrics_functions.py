import subprocess
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def install_morphometrics_packages_r():
    # Ruta al archivo de script R que instalará los paquetes
    ruta_script_r = r'Install_morphometrics.R'  # Asegúrate de que la ruta sea correcta

    # Comando para ejecutar el script de R
    command = ['Rscript', ruta_script_r]

    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")



def exploratory_morphometrics_r(
    info_data, 
    grouping_factor, 
    directorio_input, 
    output_directory,
    img_width_panel=750, img_height_panel=500, 
    img_width_stack=750, img_height_stack=500,
    nexamples=1, 
    nharmonics=10, 
    img_width_ptolemy=750, img_height_ptolemy=500,
    img_width_deviations=750, img_height_deviations=500,
    img_width_reconstructions=750, img_height_reconstructions=500,
    show=True
):
    """
    Ejecuta el script de R con los argumentos proporcionados y opcionalmente muestra las imágenes generadas.

    Parámetros:
    - info_data (str): Ruta del archivo de datos con la información a utilizar.
    - grouping_factor (str): Nombre de la columna de agrupación en info_data.
    - directorio_input (str): Directorio con las imágenes .jpg.
    - img_width_panel (int): Ancho de la imagen para el gráfico del panel.
    - img_height_panel (int): Alto de la imagen para el gráfico del panel.
    - img_width_stack (int): Ancho de la imagen para el gráfico apilado.
    - img_height_stack (int): Alto de la imagen para el gráfico apilado.
    - nexamples (int): Número de ejemplos para el loop en el script R.
    - nharmonics (int): Número de armónicos para las funciones de calibración.
    - img_width_ptolemy (int): Ancho de la imagen para el gráfico de Ptolemy.
    - img_height_ptolemy (int): Alto de la imagen para el gráfico de Ptolemy.
    - img_width_deviations (int): Ancho de la imagen para el gráfico de desviaciones efourier.
    - img_height_deviations (int): Alto de la imagen para el gráfico de desviaciones efourier.
    - img_width_reconstructions (int): Ancho de la imagen para el gráfico de reconstrucciones efourier.
    - img_height_reconstructions (int): Alto de la imagen para el gráfico de reconstrucciones efourier.
    - show (bool): Si es True, mostrará las imágenes generadas usando matplotlib.
    """
    
    # Ruta fija al script de R
    script_r_path = "Exploratory_analysis.R"
    
    # Crear el comando para ejecutar el script de R con los argumentos
    command = [
        'Rscript', 
        script_r_path,
        info_data, 
        grouping_factor, 
        directorio_input, 
        str(img_width_panel), 
        str(img_height_panel), 
        str(img_width_stack), 
        str(img_height_stack), 
        output_directory,  
        str(nexamples), 
        str(nharmonics), 
        str(img_width_ptolemy), 
        str(img_height_ptolemy), 
        str(img_width_deviations), 
        str(img_height_deviations), 
        str(img_width_reconstructions), 
        str(img_height_reconstructions)
    ]
    
    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # Si 'show' es True, intentar mostrar las imágenes generadas
        if show:
            # Definir las rutas de las imágenes exportadas por R
            exploratory_plots_dir = os.path.join(output_directory, 'exploratory_plots')
            panel_image_path = os.path.join(exploratory_plots_dir, 'panel_output.png')
            stack_image_path = os.path.join(exploratory_plots_dir, 'stack_output.png')
            
            # Mostrar las imágenes de panel y stack
            if os.path.exists(panel_image_path):
                img = mpimg.imread(panel_image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            
            if os.path.exists(stack_image_path):
                img = mpimg.imread(stack_image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            # Mostrar las imágenes generadas en el bucle de R (ptolemy, deviations, reconstructions)
            for i in range(1, nexamples + 1):
                ptolemy_image_path = os.path.join(exploratory_plots_dir, f"ptolemy_output_{i}.png")
                deviations_image_path = os.path.join(exploratory_plots_dir, f"deviations_efourier_output_{i}.png")
                reconstructions_image_path = os.path.join(exploratory_plots_dir, f"reconstructions_efourier_output_{i}.png")
                
                # Verificar y mostrar cada imagen generada en el loop
                for image_path in [ptolemy_image_path, deviations_image_path, reconstructions_image_path]:
                    if os.path.exists(image_path):
                        img = mpimg.imread(image_path)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")



def run_efourier_pca_morphometrics_r(ruta_outline_objects, nharmonics, output_directory, 
                                      img_width_boxplot=1000, img_height_boxplot=1000, 
                                      img_width_pca=1000, img_height_pca=1000, show=False, normalize="FALSE", start_point="FALSE", allign_x="TRUE"):
    """
    Ejecuta el script de R "efourier_morphometrics.R" con los argumentos proporcionados.

    Parámetros:
    - ruta_outline_objects (str): Ruta al archivo RDS con los outlines.
    - nharmonics (int): Número de armónicos para el análisis de Fourier.
    - output_directory (str): Directorio de salida donde se guardarán los resultados.
    - img_width_boxplot (int): Ancho de la imagen del boxplot.
    - img_height_boxplot (int): Alto de la imagen del boxplot.
    - img_width_pca (int): Ancho de la imagen para el gráfico PCA.
    - img_height_pca (int): Alto de la imagen para el gráfico PCA.
    - show (bool): Si es True, muestra el gráfico PCA generado con matplotlib.
    """

    # Verificar que el directorio de salida existe, si no, crearlo
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Ruta del script R
    script_r_path = "efourier_morphometrics.R"

    # Crear el comando para ejecutar el script R con los argumentos
    command = [
        'Rscript', 
        script_r_path, 
        str(ruta_outline_objects),  # Ruta al archivo RDS con los outlines
        str(nharmonics),       # Número de armónicos
        str(output_directory),      # Directorio de salida
        str(img_width_boxplot),  # Ancho del boxplot
        str(img_height_boxplot), # Alto del boxplot
        str(img_width_pca),     # Ancho de la imagen PCA
        str(img_height_pca),
        str(normalize),
        str(start_point),
        str(allign_x)
    ]

    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # Si 'show' es True, intentar mostrar el gráfico PCA
        if show:
            pca_image_path = os.path.join(output_directory, "efourier_results", "pca_output.png")
            if os.path.exists(pca_image_path):
                img = mpimg.imread(pca_image_path)
                plt.imshow(img)
                plt.axis('off')  # Desactivar los ejes
                plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")




def run_plot_pca_morphometrics_r(ruta_pca_objects, output_directory, img_width_pca=1000, img_height_pca=1000, 
                                 grouping_factor="", PC_axis1=1, PC_axis2=2, 
                                 chull_layer="FALSE", chullfilled_layer="FALSE", show=True):
    """
    Ejecuta el script de R "plot_pca_morphometrics.R" con los argumentos proporcionados.

    Parámetros:
    - ruta_pca_objects (str): Ruta al archivo RDS con el objeto PCA.
    - output_directory (str): Directorio de salida donde se guardarán los resultados.
    - img_width_pca (int): Ancho de la imagen para el gráfico PCA.
    - img_height_pca (int): Alto de la imagen para el gráfico PCA.
    - grouping_factor (str): Factor de agrupamiento opcional para la visualización PCA.
    - PC_axis1 (int): Eje principal de la PCA en el gráfico (por defecto, 1).
    - PC_axis2 (int): Eje secundario de la PCA en el gráfico (por defecto, 2).
    - chull_layer (str): Si es "TRUE", añade una capa convex hull.
    - chullfilled_layer (str): Si es "TRUE", añade una capa convex hull llena.
    - show (bool): Si es True, muestra el gráfico PCA generado con matplotlib.
    """

    # Verificar que el directorio de salida existe, si no, crearlo
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Ruta del script R
    script_r_path = "plot_pca_morphometrics.R"

    # Crear el comando para ejecutar el script R con los argumentos
    command = [
        'Rscript', 
        script_r_path, 
        str(ruta_pca_objects),      # Ruta al archivo RDS con el objeto PCA
        str(output_directory),      # Directorio de salida
        str(img_width_pca),         # Ancho de la imagen PCA
        str(img_height_pca),        # Alto de la imagen PCA
        str(grouping_factor),       # Factor de agrupamiento opcional
        str(PC_axis1),              # Eje PC1
        str(PC_axis2),                      # Eje PC2
        str(chull_layer),           # Capa de convex hull
        str(chullfilled_layer)      # Capa de convex hull llena
    ]

    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # Si 'show' es True, intentar mostrar el gráfico PCA
        pca_image_path = os.path.join(output_directory, "efourier_results", "pca_plot.png")
        if show and os.path.exists(pca_image_path):
            img = mpimg.imread(pca_image_path)
            plt.imshow(img)
            plt.axis('off')  # Desactivar los ejes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")


def run_kmeans_efourier_r(ruta_pca_objects, output_directory, max_clusters, img_width_pca=1000, img_height_pca=1000,
                          plot_xlim=250, plot_ylim=250, show=True):
    """
    Ejecuta el script de R "kmeans_Efourier_morphometric.R" con los argumentos proporcionados.

    Parámetros:
    - ruta_pca_objects (str): Ruta al archivo RDS con el objeto PCA.
    - output_directory (str): Directorio de salida donde se guardarán los resultados.
    - max_clusters (int): Número máximo de clusters a utilizar en k-means.
    - img_width_pca (int): Ancho de la imagen para el gráfico PCA.
    - img_height_pca (int): Alto de la imagen para el gráfico PCA.
    - plot_xlim (int): Límite del gráfico en el eje X.
    - plot_ylim (int): Límite del gráfico en el eje Y.
    - show (bool): Si es True, muestra el gráfico generado con matplotlib.
    """
    
    # Verificar que el directorio de salida exista, si no, crearlo
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Ruta del script R
    script_r_path = "kmeans_efourier_morphometrics.R"
    
    # Crear el comando para ejecutar el script R con los argumentos
    command = [
        'Rscript',
        script_r_path,
        str(ruta_pca_objects),      # Ruta al archivo RDS con el objeto PCA
        str(output_directory),      # Directorio de salida
        str(img_width_pca),         # Ancho de la imagen PCA
        str(img_height_pca),        # Alto de la imagen PCA
        str(max_clusters),          # Número máximo de clusters
        str(plot_xlim),             # Límite en X
        str(plot_ylim),           # Límite en Y
    ]

    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # Si 'show' es True, intentar mostrar los gráficos generados
        
        # Nombre de la imagen final del mosaico
        nombre_mosaico = os.path.join(output_directory,"kmeans_results",'kmeans_shape_plot.jpg')
        ruta_carpeta = os.path.join(output_directory, "kmeans_results")
        
        # Obtener y ordenar las imágenes por su nombre
        imagenes = sorted(os.listdir(ruta_carpeta))

        # Cargar todas las imágenes en un diccionario organizado por escenario (k)
        imagenes_dict = {}

        # Expresión regular para verificar el formato del nombre de archivo
        pattern = r"centroides_k(\d+)_cluster_(\d+)\.jpg"

        for imagen in imagenes:
            if imagen.endswith('.jpg'):
                # Usar la expresión regular para extraer los números de k y y
                match = re.match(pattern, imagen)
                if match:
                    k = int(match.group(1))  # Obtiene el número de clusters (k)
                    y = int(match.group(2))  # Obtiene el número del cluster (y)
                    
                    # Añadir la imagen al diccionario
                    if k not in imagenes_dict:
                        imagenes_dict[k] = []
                    imagenes_dict[k].append((y, os.path.join(ruta_carpeta, imagen)))
                else:
                    print(f"Advertencia: archivo {imagen} no cumple con el formato esperado.")
                    continue

        # Ordenar los clusters dentro de cada k
        for k in imagenes_dict:
            imagenes_dict[k].sort()  # Ordena las imágenes por el índice y dentro de cada k

        # Cargar las imágenes y calcular dimensiones del mosaico
        imagenes_cargadas = {k: [Image.open(imagen[1]) for imagen in imagenes_dict[k]] for k in imagenes_dict}
        ancho, alto = imagenes_cargadas[1][0].size  # Asumimos que todas las imágenes tienen el mismo tamaño
        alto_total = sum([alto for k in imagenes_cargadas])  # Altura total del mosaico
        ancho_maximo = max([ancho * len(imagenes_cargadas[k]) for k in imagenes_cargadas])  # Ancho máximo del mosaico

        # Crear imagen en blanco para el mosaico
        mosaico = Image.new('RGB', (ancho_maximo, alto_total), (255, 255, 255))

        # Colocar las imágenes en el mosaico
        y_offset = 0
        for k in sorted(imagenes_cargadas):
            x_offset = 0
            for img in imagenes_cargadas[k]:
                mosaico.paste(img, (x_offset, y_offset))
                x_offset += ancho
            y_offset += alto

        # Guardar
        mosaico.save(nombre_mosaico)
        

        # Eliminar las imágenes utilizadas
        for k in imagenes_dict:
            for _, ruta_imagen in imagenes_dict[k]:
                os.remove(ruta_imagen)  # Elimina cada imagen utilizada en el mosaico
                print(f"Eliminada: {ruta_imagen}")

        
        
        if show and os.path.exists(nombre_mosaico):
            img = mpimg.imread(nombre_mosaico)
            plt.imshow(img)
            plt.axis('off')  # Desactivar los ejes
            plt.show()

        pca_image_path = os.path.join(output_directory, "kmeans_results", "Elbow_method_plot.jpg")
        if show and os.path.exists(pca_image_path):
            img = mpimg.imread(pca_image_path)
            plt.imshow(img)
            plt.axis('off')  # Desactivar los ejes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")




def run_obtain_kmeans_classification_r(ruta_pca_objects, output_directory, img_width=750, img_height=500, 
                                       ruta_kmeans_objects="", PC_axis1=1, PC_axis2=2, 
                                       chull_layer="FALSE", chullfilled_layer="FALSE", show=True):
    """
    Ejecuta el script de R "Obtain_kmeans_classification.R" con los argumentos proporcionados.

    Parámetros:
    - ruta_pca_objects (str): Ruta al archivo RDS con el objeto PCA.
    - output_directory (str): Directorio de salida donde se guardarán los resultados.
    - img_width (int): Ancho de la imagen para el gráfico de clustering.
    - img_height (int): Alto de la imagen para el gráfico de clustering.
    - ruta_kmeans_objects (str): Ruta al archivo RDS con el objeto de clustering K-means.
    - PC_axis1 (int): Eje principal de la PCA en el gráfico (por defecto, 1).
    - PC_axis2 (int): Eje secundario de la PCA en el gráfico (por defecto, 2).
    - chull_layer (str): Si es "TRUE", añade una capa convex hull.
    - chullfilled_layer (str): Si es "TRUE", añade una capa convex hull llena.
    - show (bool): Si es True, muestra el gráfico de clustering generado con matplotlib.
    """

    # Verificar que el directorio de salida existe, si no, crearlo
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Ruta del script R
    script_r_path = "Obtain_kmeans_classification.R"

    # Crear el comando para ejecutar el script R con los argumentos
    command = [
        'Rscript', 
        script_r_path, 
        str(ruta_pca_objects),      # Ruta al archivo RDS con el objeto PCA
        str(output_directory),      # Directorio de salida
        str(img_width),             # Ancho de la imagen
        str(img_height),            # Alto de la imagen
        str(ruta_kmeans_objects),   # Ruta al archivo RDS del objeto kmeans
        str(PC_axis1),              # Eje PC1
        str(PC_axis2),              # Eje PC2
        str(chull_layer),           # Capa de convex hull
        str(chullfilled_layer)      # Capa de convex hull llena
    ]

    # Ejecutar el comando con subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Mostrar la salida del comando R
        print("Salida del comando R:")
        print(result.stdout)

        # Mostrar cualquier error si ocurre
        if result.stderr:
            print("Error:")
            print(result.stderr)

        # Si 'show' es True, intentar mostrar el gráfico de clustering
        clustered_image_path = os.path.join(output_directory, "kmeans_results", "pca_plot_clustered.png")
        if show and os.path.exists(clustered_image_path):
            img = mpimg.imread(clustered_image_path)
            plt.imshow(img)
            plt.axis('off')  # Desactivar los ejes
            plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script R: {e.stderr}")









def process_images_and_perform_pca(directory, working_directory, n_components=50, k_max=10, std_multiplier=2):
    # Step 1: Load images and convert them to binary arrays
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
    
    images = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).convert('1')  # Convert to binary
        image_array = np.array(image)  # Convert image to numpy array
        image_array = np.invert(image_array)  # Invert the binary image
        images.append(image_array)
    
    # Step 2: Flatten the list of images to a matrix of shape (k, m*n), where k is the number of images
    images = np.array(images)
    flattened_images = images.reshape(images.shape[0], -1)  # Flatten the images
    print("Flattened image matrix shape:", flattened_images.shape)

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_images = pca.fit_transform(flattened_images)

    # Explained variance for each component
    explained_variance = pca.explained_variance_ratio_

    # Create a DataFrame with the PCA values
    df_pca = pd.DataFrame(
        pca_images[:, :10],  # Take only the first 10 principal components
        columns=[f"PC{i+1}" for i in range(10)],  # Column names (PC1 to PC10)
        index=image_files  # Use actual image filenames as index
    )

    # Save the PCA values DataFrame as a TXT file
    pca_output_file = os.path.join(working_directory, "pca_values.txt")
    df_pca.to_csv(pca_output_file, sep="\t", index=True)
    print(f"PCA values file saved as {pca_output_file}")

    # Save the explained variance in another TXT file
    variance_output_file = os.path.join(working_directory, "explained_variance.txt")
    with open(variance_output_file, "w") as f:
        f.write("Principal Component\tExplained Variance\n")
        for i, var in enumerate(explained_variance, 1):
            f.write(f"PC{i}\t{var:.6f}\n")
    
    print(f"Explained variance file saved as {variance_output_file}")

    # Step 3: Calculate the mean shape in the original space
    mean_shape = pca.mean_.reshape(images.shape[1], images.shape[2])

    # For each principal component (PC1 to PC10)
    for pc in range(10):  # Iterate over the first 10 PCs
        std_pc = np.sqrt(pca.explained_variance_[pc])  # Standard deviation of the component
        direction_pc = pca.components_[pc].reshape(images.shape[1], images.shape[2])  # Direction of the PC in the original space

        # Calculate the adjusted shapes based on mean shape and standard deviation
        shape_pos = mean_shape + std_multiplier * std_pc * direction_pc  # Mean + std_multiplier*std
        shape_neg = mean_shape - std_multiplier * std_pc * direction_pc  # Mean - std_multiplier*std

        # Create a figure with 3 images: (-std_multiplier std, mean, +std_multiplier std)
        plt.figure(figsize=(12, 4))

        # Mean shape - std_multiplier std
        plt.subplot(1, 3, 1)
        plt.imshow(shape_neg > 0.5, cmap="Wistia")  # Threshold to binarize
        plt.title(f"PC{pc+1}: Mean - {std_multiplier}*std")
        plt.axis("off")

        # Mean shape
        plt.subplot(1, 3, 2)
        plt.imshow(mean_shape > 0.5, cmap="Wistia")  # Threshold to binarize
        plt.title(f"PC{pc+1}: Mean")
        plt.axis("off")

        # Mean shape + std_multiplier std
        plt.subplot(1, 3, 3)
        plt.imshow(shape_pos > 0.5, cmap="Wistia")  # Threshold to binarize
        plt.title(f"PC{pc+1}: Mean + {std_multiplier}*std")
        plt.axis("off")

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(working_directory, f"pc{pc+1}_influence.jpg"), format="jpg")
        plt.show()

    # Step 4: Evaluate KMeans for different values of k (1 to k_max)
    distortions = []  # To store inertia for each k
    for k in range(1, k_max):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_images)
        distortions.append(kmeans.inertia_)

        # Visualize the centroids of each cluster
        centroids_pca = kmeans.cluster_centers_

        # Project centroids back to original space
        centroids_original = pca.inverse_transform(centroids_pca)

        # Visualize centroids
        rows = (k + 4) // 5  # Calculate number of rows
        cols = min(k, 5)  # Limit to 5 columns per row (maximum)

        plt.figure(figsize=(15, 3 * rows))  # Adjust figure size based on rows

        for i, centroid in enumerate(centroids_original):
            # Convert the centroid (vector) to a binary image
            binary_image = centroid.reshape(images.shape[1], images.shape[2]) > 0.5  # Binarize with threshold

            # Create a subplot
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(binary_image, cmap="Wistia")  # Use a brown colormap
            ax.set_title(f'Centroid {i + 1}')
            ax.axis('off')

        # Save centroid image as JPG
        plt.tight_layout()
        plt.savefig(os.path.join(working_directory, f"centroids_k_{k}.jpg"), format="jpg")
        plt.show()

    # Step 5: Evaluate optimal number of clusters using the elbow method
    plt.plot(range(1, k_max), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Distortion)')
    plt.title('Elbow Method for Selecting k')
    plt.savefig(os.path.join(working_directory, "elbow_plot.jpg"), format="jpg")
    plt.show()
