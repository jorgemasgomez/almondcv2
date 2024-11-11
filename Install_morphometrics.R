# Morphometrics install packages

# Especificar el repositorio de CRAN
repos <- "https://cloud.r-project.org/"

# Comprobar si el paquete "Momocs" está instalado
if (!requireNamespace("Momocs", quietly = TRUE)) {
  cat("El paquete 'Momocs' no está instalado. Instalando...\n")
  install.packages("Momocs", repos = repos)
} else {
  cat("El paquete 'Momocs' ya está instalado.\n")
}

# Comprobar si el paquete "dplyr" está instalado
if (!requireNamespace("dplyr", quietly = TRUE)) {
  cat("El paquete 'dplyr' no está instalado. Instalando...\n")
  install.packages("dplyr", repos = repos)
} else {
  cat("El paquete 'dplyr' ya está instalado.\n")
}