#E_fourier_morphometrics
library(Momocs)
# Cargar el objeto PCA desde el archivo .rds

args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 10) {
ruta_outline_objects<-args[1]
nharmonics<-as.numeric(args[2])
output_directory <- args[3] 
img_width_boxplot <- as.numeric(args[4])    # Ancho de la imagen para el panel
img_height_boxplot <- as.numeric(args[5])
img_width_pca <- as.numeric(args[6])    # Ancho de la imagen para el panel
img_height_pca <- as.numeric(args[7])
normalize <- as.logical(args[8])
start_point <- as.logical(args[9])
allign_x <- as.logical(args[10])

} else {
  stop("No se pasaron suficientes argumentos.")
}
output_folder <- file.path(output_directory, "efourier_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}
outlines_objects <- readRDS(ruta_outline_objects)

if (allign_x) {
  outlines_objects <- coo_slidedirection(outlines_objects, direction = "right", center = TRUE)
  outlines_objects <- coo_alignxax(outlines_objects)
}


e_fourier_contours <- efourier(outlines_objects, nb.h=nharmonics, norm = normalize, start = start_point)
write.table(e_fourier_contours$coe, 
            file = file.path(output_folder, "e_fourier_coefs.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Crear un boxplot y guardarlo como imagen PNG
png(filename = file.path(output_folder, "boxplot_output.png"), 
    width = img_width_boxplot, 
    height = img_height_boxplot)
boxplot(e_fourier_contours, drop = 1)
dev.off()

# Realizar PCA sobre los resultados de Fourier
pca_fourier <- PCA(e_fourier_contours)

# Guardar las coordenadas de los PCs en un archivo de texto
write.table(pca_fourier$x, 
            file = file.path(output_folder, "e_fourier_pcs_coordinates.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Guardar los valores propios (eigenvalues) del PCA en un archivo de texto
write.table(pca_fourier$eig, 
            file = file.path(output_folder, "e_fourier_pcs_eigenvalues.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)

# Crear un gráfico PCA y guardarlo como imagen PNG
png(filename = file.path(output_folder, "pca_output.png"), 
    width = img_width_pca, 
    height = img_height_pca)
plot_PCA(pca_fourier)
dev.off()

# Guardar el objeto PCA como archivo .rds
saveRDS(pca_fourier, file = file.path(output_folder, "pca_fourier.rds"))