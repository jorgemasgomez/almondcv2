import subprocess
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
                                      img_width_boxplot=750, img_height_boxplot=500, 
                                      img_width_pca=750, img_height_pca=500, show=False):
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
        str(img_height_pca)    # Alto de la imagen PCA
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