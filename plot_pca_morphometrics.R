#PLOT PCA

ibrary(Momocs)
# Cargar el objeto PCA desde el archivo .rds

args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 4) {
  ruta_pca_objects<-args[1]
  output_directory <- args[2] 
  img_width_pca <- as.numeric(args[3])    # Ancho de la imagen para el panel
  img_height_pca <- as.numeric(args[4])
  grouping_factor<- as.factor(args[5])
  PC_axis1 <-as.factor(args[6])
  PC_axis2 <-as.factor(args[7])

  
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


plot_PCA(pca_fourier,f = grouping_factor, axes = c(PC_axis1, PC_axis2))



dev.off()