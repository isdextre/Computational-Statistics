# =========================
# 1. Cargar librerías
# =========================
library(dplyr)

# =========================
# 2. Leer la data
# =========================
df <- read.csv("data_raw/BKB_WaterQualityData_2020084.csv", stringsAsFactors = FALSE)

# Ver primeras filas
head(df)

# Ver últimas filas
tail(df)

# Dimensiones
dim(df)

# Nombres de columnas
names(df)

# Estructura general
str(df)

# Resumen general
summary(df)