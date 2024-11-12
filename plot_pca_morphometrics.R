#PLOT PCA

library(Momocs)
# Cargar el objeto PCA desde el archivo .rds

args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 9) {
  ruta_pca_objects<-args[1]
  output_directory <- args[2] 
  img_width_pca <- as.numeric(args[3])    # Ancho de la imagen para el panel
  img_height_pca <- as.numeric(args[4])
  grouping_factor<- args[5]
  PC_axis1 <-as.numeric(args[6])
  PC_axis2 <-as.numeric(args[7])
  chull_layer<- as.logical(args[8])
  chullfilled_layer<- as.logical(args[9])
  
} else {
  stop("No se pasaron suficientes argumentos.")
}
output_folder <- file.path(output_directory, "efourier_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}
pca_fourier <- readRDS(ruta_pca_objects)


png(filename = file.path(output_folder, "pca_plot.png"), 
    width = img_width_pca, 
    height = img_height_pca)




if (!nzchar(grouping_factor)) {
  # Aquí el código para cuando grouping_factor está vacío
  print("La variable grouping_factor está vacía.")
  plot_PCA(pca_fourier,axes = c(PC_axis1, PC_axis2))
} else {
  # Aquí el código para cuando grouping_factor NO está vacío
  
  plot_PCA(pca_fourier,f = grouping_factor, axes = c(PC_axis1, PC_axis2),
           chull=chull_layer, chullfilled=chullfilled_layer)
}

dev.off()