############################################
rm(list = ls())  # Limpiar ambiente

# Instalar paquetes

install.packages(c(
  # Manipulación y limpieza de datos
  "tidyverse",      # dplyr, ggplot2, tidyr, etc.
  "skimr",          # Resúmenes estadísticos
  "janitor",        # Limpieza de nombres de columnas
  
  # Análisis estadístico y PCA
  "FactoMineR",     # PCA y análisis multivariado
  "factoextra",     # Visualización de PCA
  "corrplot",       # Matrices de correlación
  "GGally",         # Matriz de gráficos de correlación
  "Hmisc",          # Matrices de correlación con p-valores
  
  # Tests estadísticos
  "car",            # ANOVA, tests de normalidad
  "nortest",        # Tests de normalidad (Anderson-Darling, etc.)
  "psych",          # Análisis factorial, descriptivos
  
  # Visualización avanzada
  "ggplot2",        # Gráficos base
  "ggpubr",         # Arreglo de múltiples gráficos
  "patchwork",      # Combinar gráficos
  "viridis",        # Paletas de colores
  
  # Datos y utilidades
  "here",           # Rutas relativas
  "readxl",         # Leer Excel si es necesario
  "writexl"         # Exportar a Excel
))





# Verificar que todos los paquetes cargan correctamente
paquetes <- c("tidyverse", "skimr", "janitor", "FactoMineR", 
              "factoextra", "corrplot", "GGally", "Hmisc", 
              "car", "nortest", "psych", "ggpubr", "patchwork", 
              "viridis", "here", "readxl", "writexl")

for (p in paquetes) {
  if (require(p, character.only = TRUE, quietly = TRUE)) {
    cat("✅", p, "\n")
  } else {
    cat("❌", p, "- ERROR\n")
  }
}
