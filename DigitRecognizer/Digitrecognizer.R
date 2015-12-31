library(ggplot2)
library(e1071)
library(ElemStatLearn)
library(plyr)
library(class)

import.csv <- function(filename){
  return(read.csv(filename, sep = ",", header = TRUE, stringsAsFactors = FALSE))
}

train.data <- import.csv("train.csv")
test.data <- import.csv("test.csv")
# test.data <- train.data[30001:32000,]
# train.data <- train.data[1:6000,]

#Performing PCA on the dataset to reduce the dimensionality of the data

get_PCA <- function(dataset){
  dataset.cov <- cov(dataset, use = "everything")
  dataset.cov.svd <- svd(dataset.cov)
  return(dataset.cov.svd)
}

#Pre-process train

preprocess <- function(train, test){
  features.unit.variance <- names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0)])
  train <- train[,!(colnames(train) %in% features.unit.variance)]
  train.label <- data.frame(label = train$label)
  train <- train[,!(colnames(train) %in% "label")]
  norm.data <- train
  mean_sdev_train <- normalize(norm.data)
  train <- data.matrix(train)
  train <- scale(train, scale = T, center = T)
  pr.comp <- get_PCA(train)
  train <- data.frame(train %*% pr.comp$u[,1:200])
  #train <- train[,1:80]
  train <- cbind(train, train.label)
  #Project test data in the training data space
  test <- test[,!(colnames(test) %in% features.unit.variance)]
  train.mean <- mean_sdev_train$mean.data
  train.sdev <- mean_sdev_train$sdev
  #Normalize the test data according to the train data
  test = data.matrix(test)
  for(i in (1:nrow(test))){
    test[i,] = test[i,] - train.mean
    test[i,] = test[i,] / train.sdev
  }
  test <- data.frame(test %*% pr.comp$u[,1:200])
  #test <- test[,1:80]
  result <- list(train = train, test = test)
  return(result)
}

#Normalize train data
normalize <- function(data){
  mean.data = vector()
  sdev = vector()
  for(i in (1:ncol(data))){
    mean.data[i] = mean(data[,i])
    sdev[i] = sd(data[,i])
  }
  result <- list(mean.data = mean.data, sdev = sdev)
}

#Perform k-fold cross validation

do_cv_class <- function(df, k, classifier){
  num_of_nn = gsub("[^[:digit:]]","",classifier)
  classifier = gsub("[[:digit:]]","",classifier)
  if(num_of_nn == "")
  {
    classifier = c("get_pred_",classifier)
  }
  else
  {
    classifier = c("get_pred_k",classifier)
    num_of_nn = as.numeric(num_of_nn)
  }
  classifier = paste(classifier,collapse = "")
  func_name <- classifier
  output = vector()
  size_distr = c()
  n = nrow(df)
  for(i in 1:n)
  {
    a = 1 + (((i-1) * n)%/%k)
    b = ((i*n)%/%k)
    size_distr = append(size_distr, b - a + 1)
  }
  
  row_num = 1:n
  sampling = list()
  for(i in 1:k)
  {
    s = sample(row_num,size_distr)
    sampling[[i]] = s
    row_num = setdiff(row_num,s)
  }
  prediction.df = data.frame()
  outcome.list = list()
  
  for(i in 1:k)
  {
    testSample = sampling[[i]]
    train_set = df[-testSample,]
    test_set = df[testSample,]
    test_set_label = test_set$label
    test_set = test_set[,!colnames(test_set) %in% "label"]
    datasets = preprocess(train_set, test_set)
    train_set = datasets$train
    test_set = datasets$test
    if(num_of_nn == "")
    {
      classifier = match.fun(classifier)
      result = classifier(train_set,test_set)
      confusion.matrix <- table(pred = result, true = test_set_label)
      accuracy <- sum(diag(confusion.matrix)*100)/sum(confusion.matrix)
      print(confusion.matrix)
      outcome <- list(sample_ID = i, Accuracy = accuracy)
      outcome.list <- rbind(outcome.list, outcome)
    }
    else
    {
      
      classifier = match.fun(classifier)
      result = classifier(train_set,test_set)
      confusion.matrix <- table(pred = result, true = test_set_label)
      accuracy <- sum(diag(confusion.matrix)*100)/sum(confusion.matrix)
      print(confusion.matrix)
      outcome <- list(sample_ID = i, Accuracy = accuracy)
      outcome.list <- rbind(outcome.list, outcome)
    }
  }
  return(outcome.list)
}

#Support Vector Machines with linear kernel
get_pred_svm <- function(train,test){
  digit.class.train <- as.factor(train$label)
  train.features <- train[,!colnames(train) %in% "label"]
  svm.model <- svm(train.features, digit.class.train, cost = 1000, gamma = 1/ncol(train.features), kernel = "radial")
  result <- (0:9)[predict(svm.model, test)]
  return(result)
}

#KNN model
get_pred_knn <- function(train,test){
  digit.class.train <- as.factor(train$label)
  train.features <- train[,!colnames(train) %in% "label"]
  knn.model <- knn(train.features, test, digit.class.train)
  return(knn.model)
}

#Final run

get_output <- function(train_set, test_set){
  datasets = preprocess(train_set, test_set)
  train_set = datasets$train
  test_set = datasets$test
  result = get_pred_svm(train_set,test_set)
  return(result)
}

