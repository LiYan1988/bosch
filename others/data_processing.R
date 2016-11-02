## wd etc ####
library(data.table)
library(recommenderlab)
require(readr)
require(caret)

## date ####
xtrain <- read_csv('../input/train_date.csv')
xtrain[is.na(xtrain)] <- -99

flc <- findLinearCombos(xtrain[,1:100])
blacklist <- colnames(xtrain[,1:100])[flc$remove]
wx <- which(colnames(xtrain) %in% blacklist)
xtrain <- xtrain[,-wx]

for (ii in 1:25)
{
  xrange <- sample(ncol(xtrain),150)
  flc <- findLinearCombos(xtrain[,xrange])
  blacklist <- colnames(xtrain[,xrange])[flc$remove]
  
  if (length(flc$remove))
  {
    wx <- which(colnames(xtrain) %in% blacklist)
    xtrain <- xtrain[,-wx]
  }
  
  print(ncol(xtrain))
}

write_csv(xtrain, path = '../input/xtrain_date.csv')

xtest <- read_csv('../input/test_date.csv')
wcol <- which(colnames(xtest) %in% colnames(xtrain))
xtest <- xtest[,wcol]
write_csv(xtest, path = '../input/xtest_date.csv')
