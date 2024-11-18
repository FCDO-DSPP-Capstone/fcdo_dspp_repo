### first settings ----
options(scipen=999) # Desactiva la notación científica
options(max.print = 99999999) # Max print de la consola
rm(list=(ls()))   # limpia el enviroment (dispongo de m?s memoria temporal)

# Vamos a cargar algunas librer?as que vamos a utilizar
install_load <- function(packages){
  for (i in packages) {
    if (i %in% rownames(installed.packages())) {
      library(i, character.only=TRUE)
    } else {
      install.packages(i)
      library(i, character.only = TRUE)
    }
  }
}
install_load(c("rio", 
               "tidyverse", 
               "janitor", 
               "wnominate", 
               "plotly"))

## 2. data import ----
un_data <- import('data_in/un_data/2024_09_12_ga_resolutions_voting.csv')

# EEn verdad hazlo en python pero ya q estamos. 
# Corre una versión del nominate y otra PCA
# Igual copiale weas buenas al W-Nominate. 
# Al hacer el PCA, dropea las votaciones Unanimes. 
# Dsps, gira el plano resultado para q la izquierda quede a la izquierda y la derecha a la derecha. 
# Tengo q estudiar bien como codificar las variables en PCA para q quede bien 
# Calcular el PCA para todos los años con una ventana movil de 5 años. 
# MMissing data is not missing, is a kind of vote. N= -1, Abs or abt = 0, Y = 1. 




