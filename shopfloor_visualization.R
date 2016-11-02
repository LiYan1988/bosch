options(warn=-1)
setwd("C:/users/lyaa/Documents/bosch")
library(data.table)
library(dtplyr)
library(dplyr)
library(raster)

n_train <- 1183747
n_test <- 1183748
n_show <- 10
n_skip <- 0

dtrainNum <- fread("input/train_numeric.csv", skip=n_skip, nrows=10)
dtrainDat <- fread("input/train_date.csv", skip=n_skip, nrows=10)
dtrainCat <- fread("input/train_categorical.csv", skip=n_skip, nrows=10)

dtrainNumNames <- names(dtrainNum)
dtrainDatNames <- names(dtrainDat)
dtrainCatNames <- names(dtrainCat)

dtrainDatNA <- apply(dtrainDat, MARGIN = 1, FUN = function(x) length(x[is.na(x)]) )
dtrainCatNA <- apply(dtrainCat, MARGIN = 1, FUN = function(x) length(x[is.na(x)]) )
dtrainNumNA <- apply(dtrainNum, MARGIN = 1, FUN = function(x) length(x[is.na(x)]) )