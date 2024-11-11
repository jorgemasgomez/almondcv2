# Cargar paquetes necesarios
library(Momocs)
library(dplyr)

# Capturar los argumentos pasados al script
args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 16) {
  info_data <- args[1]               # Archivo con los datos
  grouping_factor <- args[2]         # Factor de agrupación
  directorio <- args[3]              # Directorio donde están los archivos .jpg
  img_width_panel <- as.numeric(args[4])    # Ancho de la imagen para el panel
  img_height_panel <- as.numeric(args[5])   # Alto de la imagen para el panel
  img_width_stack <- as.numeric(args[6])    # Ancho de la imagen para el stack
  img_height_stack <- as.numeric(args[7])   # Alto de la imagen para el stack
  output_directory <- args[8]        # Directorio de salida para las imágenes
  nexamples <- as.numeric(args[9])   # Número de ejemplos para el loop
  nharmonics <- as.numeric(args[10]) # Número de armónicos para las funciones calibrate
  img_width_ptolemy <- as.numeric(args[11])    # Ancho de la imagen para Ptolemy
  img_height_ptolemy <- as.numeric(args[12])   # Alto de la imagen para Ptolemy
  img_width_deviations <- as.numeric(args[13]) # Ancho de la imagen para calibrate_deviations_efourier
  img_height_deviations <- as.numeric(args[14])# Alto de la imagen para calibrate_deviations_efourier
  img_width_reconstructions <- as.numeric(args[15]) # Ancho de la imagen para calibrate_reconstructions_efourier
  img_height_reconstructions <- as.numeric(args[16])# Alto de la imagen para calibrate_reconstructions_efourier
} else {
  stop("No se pasaron suficientes argumentos.")
}

# Comprobar si el archivo info_data existe y no está vacío
if (nzchar(info_data) && file.exists(info_data) && file.info(info_data)$size > 0) {
  # Si el archivo existe y no está vacío, cargarlo
  info_data_df <- read.table(info_data, sep = "\t", header = TRUE)
  info_data_df <- info_data_df %>% mutate_all(as.factor)
} else {
  # Si info_data no es válido, eliminar la variable
  cat("El archivo info_data no es válido o está vacío. Se omitirá el uso de esta variable.\n")
  rm(info_data)  # Eliminar la variable info_data
}

# Crear la carpeta exploratory_plots dentro de output_directory si no existe
output_folder <- file.path(output_directory, "exploratory_plots")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Crear el vector con las rutas de los archivos .jpg en la carpeta
jpg_path_list <- list.files(path = directorio, pattern = "\\.jpg$", full.names = TRUE)
outlines_objects <- import_jpg(jpg.paths = jpg_path_list)

# Establecer outlines_objects con fac si info_data_df existe
if (exists("info_data_df")) {
  outlines_objects <- Out(x = outlines_objects, fac = info_data_df)
  
  # Guardar la imagen del panel
  png(filename = file.path(output_folder, "panel_output.png"), width = img_width_panel, height = img_height_panel)
  panel(outlines_objects, fac = info_data_df[[grouping_factor]])
  dev.off()
  
  # Guardar la imagen del stack
  png(filename = file.path(output_folder, "stack_output.png"), width = img_width_stack, height = img_height_stack)
  stack(outlines_objects)
  dev.off()
  
} else {
  outlines_objects <- Out(x = outlines_objects)
  
  # Guardar la imagen del panel
  png(filename = file.path(output_folder, "panel_output.png"), width = img_width_panel, height = img_height_panel)
  panel(outlines_objects)
  dev.off()
  
  # Guardar la imagen del stack
  png(filename = file.path(output_folder, "stack_output.png"), width = img_width_stack, height = img_height_stack)
  stack(outlines_objects)
  dev.off()
}

# Loop para ejecutar Ptolemy, calibrate_deviations_efourier y calibrate_reconstructions_efourier
for (i in 1:nexamples) {
  # Seleccionar un índice aleatorio dentro de outlines_objects
  random_index <- sample(1:length(outlines_objects), 1)
  
  
  # Guardar la imagen de Ptolemy con tamaño personalizado
  png(filename = file.path(output_folder, paste0("ptolemy_output_", i, ".png")), width = img_width_ptolemy, height = img_height_ptolemy)
  Ptolemy(outlines_objects[random_index], nb.h = nharmonics)
  dev.off()
  
  # Guardar la imagen de calibrate_deviations_efourier con tamaño personalizado
  png(filename = file.path(output_folder, paste0("deviations_efourier_output_", i, ".png")), width = img_width_deviations, height = img_height_deviations)
  calibrate_deviations_efourier(outlines_objects, range = 1:nharmonics, plot = TRUE)
  dev.off()
  
  # Guardar la imagen de calibrate_reconstructions_efourier con tamaño personalizado
  png(filename = file.path(output_folder, paste0("reconstructions_efourier_output_", i, ".png")), width = img_width_reconstructions, height = img_height_reconstructions)
  print(calibrate_reconstructions_efourier(outlines_objects, range = c(1:nharmonics)))
  dev.off()
}