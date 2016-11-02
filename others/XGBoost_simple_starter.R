# Author: Haozhen Wu
# Date: 8/28/2016

# XGBosst simple starter
# use 300+ important features selected by H2O GBM
package_list <- c("data.table","Matrix","xgboost")
for(pkg in package_list){
  if (!require(pkg,character.only = T)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg,character.only = T)
  }
  library(pkg,character.only = T)
}
setwd("~/kaggle/Bosch")

training_number <- 200000
label <- fread("./train_numeric.csv",select = c("Response"),nrow = training_number)
importance = fread("./imp_matrix_H2O_GBM_allFeatures.csv")

#----- numeric features
dt_train_numeric = fread("./train_numeric.csv",select = importance[,variable],nrow = training_number)
# assign 0 to numeric missing
replace_num_na <- function(col){
  col[col==""|is.na(col)]= 0
  return(col)
}
for(i in seq_along(colnames(dt_train_numeric))){
  dt_train_numeric[,(colnames(dt_train_numeric)[i]):= replace_num_na(dt_train_numeric[[colnames(dt_train_numeric)[i]]])]
  dt_train_numeric[,(colnames(dt_train_numeric)[i]):=as.numeric(dt_train_numeric[[colnames(dt_train_numeric)[i]]])] 
}

#----- categorical features
dt_train_categoical <- fread("./train_categorical.csv",select = importance[,variable],nrow = training_number)
#replace empty value by "missing"
replace_char_na <- function(col){
  col[which(col==""|is.na(col))]= "missing"
  return(col)
}
for(i in seq_along(colnames(dt_train_categoical))){
  dt_train_categoical[,(colnames(dt_train_categoical)[i]):=replace_char_na(dt_train_categoical[[colnames(dt_train_categoical)[i]]])]
  dt_train_categoical[,(colnames(dt_train_categoical)[i]):=as.factor(dt_train_categoical[[colnames(dt_train_categoical)[i]]])] 
}

#------ date features
dt_train_date <- fread("./train_date.csv",select = importance[,variable],nrow = training_number)
for(i in seq_along(colnames(dt_train_date))){
  dt_train_date[,(colnames(dt_train_date)[i]):= replace_num_na(dt_train_date[[colnames(dt_train_date)[i]]])]
  dt_train_date[,(colnames(dt_train_date)[i]):=as.numeric(dt_train_date[[colnames(dt_train_date)[i]]])] 
}

#----- combine and make xgb data
train_combine <- cbind(dt_train_numeric,dt_train_categoical,dt_train_date)
rm(dt_train_numeric,dt_train_categoical,dt_train_date)
gc()
train <- sparse.model.matrix(~.-1,data=train_combine)
dtrain <- xgb.DMatrix(data = train,label = label[,Response])

set.seed(2016)
param <- list(objective = "binary:logistic"
              ,eta = 0.1
              ,max.depth = 5
              ,min_child_weight = 10
              ,max_delta_step = 5
              ,subsample = 0.7
              ,colsample_bytree = 0.7
              # ,scale_pos_weight = table(label)[1]/table(label)[2]
)
round <- 100
xgbtrain <- xgb.train(data = dtrain, params=param, nrounds = round)
# 3 fold cross vaildation achieved 0.712 based on this setting
rm(dtrain,train,train_combine)
gc()

#--------------------------make prediction

#----- numeric features 
dt_test_numeric <- fread("./test_numeric.csv",select = importance[,variable])

# assign 0 to numeric missing
for(i in seq_along(colnames(dt_test_numeric))){
  dt_test_numeric[,(colnames(dt_test_numeric)[i]):= replace_num_na(dt_test_numeric[[colnames(dt_test_numeric)[i]]])]
  dt_test_numeric[,(colnames(dt_test_numeric)[i]):=as.numeric(dt_test_numeric[[colnames(dt_test_numeric)[i]]])] 
}

#----- categorical features
dt_test_categoical <- fread("./test_categorical.csv",select = importance[,variable])

#replace empty value by "missing"
for(i in seq_along(colnames(dt_test_categoical))){
  dt_test_categoical[,(colnames(dt_test_categoical)[i]):=replace_char_na(dt_test_categoical[[colnames(dt_test_categoical)[i]]])]
  dt_test_categoical[,(colnames(dt_test_categoical)[i]):=as.factor(dt_test_categoical[[colnames(dt_test_categoical)[i]]])] 
}

#------ date features
dt_test_date <- fread("./test_date.csv",select = importance[,variable])
for(i in seq_along(colnames(dt_test_date))){
  dt_test_date[,(colnames(dt_test_date)[i]):= replace_num_na(dt_test_date[[colnames(dt_test_date)[i]]])]
  dt_test_date[,(colnames(dt_test_date)[i]):=as.numeric(dt_test_date[[colnames(dt_test_date)[i]]])] 
}
#------ combine
test_combine <- cbind(dt_test_numeric,dt_test_categoical,dt_test_date)
rm(dt_test_numeric,dt_test_categoical,dt_test_date)
test <- sparse.model.matrix(~.-1,data=test_combine)
gc()

pred <- predict(xgbtrain,test)
Id <- fread("./test_numeric.csv",select = "Id")

result <- data.table(Id = Id[,Id],Response = pred)
result <- result[order(pred,decreasing = T)]
# Naively choose top 5500 instance as label 1, you can change the threshold by yourself.
result[1:5500,Response:=1]
result[5501:nrow(result),Response:=0]

write.csv(result,paste0("./xgb_200k_Sample.csv"),row.names=F)
