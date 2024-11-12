
library(Momocs)
# Cargar el objeto PCA desde el archivo .rds

args <- commandArgs(trailingOnly = TRUE)

# Comprobar si se han pasado suficientes argumentos
if (length(args) >= 7) {
  ruta_pca_objects<-args[1]
  output_directory <- args[2] 
  img_width <- as.numeric(args[3])    # Ancho de la imagen para el panel
  img_height <- as.numeric(args[4])
  max_clusters <- as.numeric(args[5])
  plot_xlim<-as.numeric(args[6])
  plot_ylim<-as.numeric(args[7])
  

} else {
  stop("No se pasaron suficientes argumentos.")
}
output_folder <- file.path(output_directory, "kmeans_results")
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}
pca_fourier <- readRDS(ruta_pca_objects)


# Solicitar al usuario el número máximo de clusters


# Inicializamos la lista para almacenar las formas reconstruidas de los centroides
centroid_shapes <- list()
wss_values <- numeric(max_clusters)

# Loop para realizar kmeans para cada número de clusters desde 1 hasta el máximo
for (num_clusters in 1:max_clusters) {
  
  # Realizar kmeans con el número actual de clusters
  kmeans_result <- kmeans(pca_fourier$x, centers=num_clusters)
  saveRDS(pca_fourier, file = file.path(output_folder, paste0("pca_fourier_",num_clusters,".rds")))
  wss_values[num_clusters] <- kmeans_result$tot.withinss  # Suma de los errores cuadrados dentro de los clusters
  centroids_pca <- kmeans_result$centers
  
  # Inicializamos la lista para almacenar las formas reconstruidas de los centroides
  centroid_shapes <- list()
  
  # Proyectar cada centroide al espacio de Fourier
  for (i in 1:num_clusters) {
    # Centroides en el espacio PCA
    projections <- as.matrix(centroids_pca[i, , drop = FALSE])  # Convertimos a matriz
    projections <- t(projections)
    
    # Proyectamos al espacio de Fourier y añadimos la media (mshape)
    reconstructed_coef <- as.vector(pca_fourier$rotation %*% projections) + pca_fourier$mshape
    
    # Número de armónicos (ajusta según tu caso)
    nb_h <- 10  # Cambia este valor si usaste un número diferente de armónicos
    
    # Divide los coeficientes reconstruidos en an, bn, cn, dn
    an <- reconstructed_coef[1:nb_h]
    bn <- reconstructed_coef[(nb_h + 1):(2 * nb_h)]
    cn <- reconstructed_coef[(2 * nb_h + 1):(3 * nb_h)]
    dn <- reconstructed_coef[(3 * nb_h + 1):(4 * nb_h)]
    
    # Usa el centro original de la forma para mantener el desplazamiento
    ao <- pca_fourier$center[1]
    co <- pca_fourier$center[2]
    
    # Crea un objeto efourier con los coeficientes reconstruidos
    ef_centroid <- list(an = an, bn = bn, cn = cn, dn = dn, ao = ao, co = co)
    
    # Reconstruye la forma y almacénala en la lista
    centroid_shapes[[i]] <- efourier_i(ef_centroid)
  }
  
  
  
  # Visualiza las formas reconstruidas de los centroides y guarda la imagen
  for (i in 1:num_clusters) {
    # Crear el nombre del archivo para guardar la imagen
    output_file <- paste0("centroides_k", num_clusters,"_cluster_", i, ".jpg")
    
    # Guardar cada imagen como un archivo .jpg
    jpeg(filename = file.path(output_folder,output_file), width = img_width, height = img_height)
    coo_plot(centroid_shapes[[i]], border='orange3', col = 'orange3',
             xy.axis = FALSE, xlim = c(-plot_xlim, plot_xlim), ylim = c(-plot_ylim, plot_ylim))  # Muestra cada centroide reconstruido
    text(0, 0, paste("k =", i), col="white", cex=1)  # Añade la etiqueta "k=1", "k=2", etc.
    dev.off()
    cat("Imagen de los centroides con", num_clusters, "clusters guardada como", output_file, "\n")
  }
  
}
output_file <- paste0("Elbow_method_plot.jpg")
jpeg(filename = file.path(output_folder,output_file), width = img_width, height = img_height)
plot(1:max_clusters, wss_values, type="b", pch=19, col="blue", 
     xlab="Número de clusters", ylab="Suma de Errores Cuadrados Dentro de los Clusters (WSS)",
     main="Método del Codo (Elbow Method)")
dev.off()