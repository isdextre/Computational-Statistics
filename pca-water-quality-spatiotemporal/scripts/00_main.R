# Cargar librerías
library(tidyverse)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(skimr)


datos <- read.csv("C:/Users/JHOSSEP/Documents/REPO/Computational-Statistics/pca-water-quality-spatiotemporal/data/data_raw/BKB_WaterQualityData_2020084.csv", fileEncoding = "UTF-8")

# Verificar que se cargó bien
cat("Datos cargados:", nrow(datos), "filas,", ncol(datos), "columnas\n")

head(datos)
summary(datos)
