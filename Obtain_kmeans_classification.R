
# Obtain clustering classification

library(Momocs)
library(dplyr)

# Cargar el objeto PCA desde el archivo .rds

args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 9) {
  ruta_pca_objects<-args[1]
  output_directory <- args[2] 
  img_width <- as.numeric(args[3])    # Ancho de la imagen para el panel
  img_height <- as.numeric(args[4])
  ruta_kmeans_objects<-args[5]
  PC_axis1 <-as.numeric(args[6])
  PC_axis2 <-as.numeric(args[7])
  chull_layer<- as.logical(args[8])
  chullfilled_layer<- as.logical(args[9])
  
  
} else {
  stop("No se pasaron suficientes argumentos.")
}

print("hello")


output_folder <- file.path(output_directory, "kmeans_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

pca_fourier <- readRDS(ruta_pca_objects)
kmeans_result<-readRDS(ruta_kmeans_objects)
nb_h <- as.numeric(length(pca_fourier$eig)/4)


write.table(kmeans_result$cluster, 
            file = file.path(output_folder, "cluster_classification.txt"), 
            sep = "\t", 
            row.names = TRUE, 
            col.names = TRUE, 
            quote = FALSE)


print(names(kmeans_result$cluster))
# Convertir el vector nombrado kmeans_result$cluster en un data frame
cluster_df <- tibble(Binary_mask_picture = names(kmeans_result$cluster), 
                     Cluster = as.integer(kmeans_result$cluster))

# Combinar con pca_fourier$fac usando un left join para añadir la columna de Cluster
pca_fourier$fac <- pca_fourier$fac %>%
  left_join(cluster_df, by = "Binary_mask_picture")


png(filename = file.path(output_folder, "pca_plot_clustered.png"), 
    width = img_width, 
    height = img_height)
plot_PCA(pca_fourier,f = "Cluster", axes = c(PC_axis1, PC_axis2),
         chull=chull_layer, chullfilled=chullfilled_layer)
dev.off()
