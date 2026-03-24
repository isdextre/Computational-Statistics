read_and_clean_water_data <- function(path) { -- lee un path
  df <- read.csv(path, stringsAsFactors = FALSE)
  df[df == "N/A"] <- NA

  cols_keep <- c(
    "Site_Id",
    "Read_Date",
    "Salinity..ppt.",
    "Dissolved.Oxygen..mg.L.",
    "pH..standard.units.",
    "Secchi.Depth..m.",
    "Water.Depth..m.",
    "Water.Temp..C.",
    "AirTemp..C."
  )

  df <- df[, cols_keep]
  df$Read_Date <- as.Date(df$Read_Date, format = "%m/%d/%Y")

  numeric_cols <- c(
    "Salinity..ppt.",
    "Dissolved.Oxygen..mg.L.",
    "pH..standard.units.",
    "Secchi.Depth..m.",
    "Water.Depth..m.",
    "Water.Temp..C.",
    "AirTemp..C."
  )

  df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)
  df <- na.omit(df)

  return(df)
}