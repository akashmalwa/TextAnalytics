library(Matrix)

# Read data from the file into a dataframe
readData <- function(filename)
{
  conn = file(filename, open = "r")
  docVect = readLines(conn)
  doc_word_vectors = data.frame(docVect, stringsAsFactors = FALSE)
  close(conn)
  return(doc_word_vectors)
}

# Translate the document vectors into a sparse matrix
TermDocMatrix <- function(docVect.df)
{
  # N = number of documents
  N = nrow(docVect.df)
  y <- data.frame(matrix(0, N, 2))
  colnames(y) <- c("DocID", "Category")
  data.df <- data.frame()
  for(i in 1:N)
  {
    doc <- docVect.df[i,1]
    doc.df <- data.frame(unlist(strsplit(doc, " ")), stringsAsFactors = F)
    y[i,1] = i
    y[i,2] = as.numeric(doc.df[1,1])
    doc.df <- data.frame(doc.df[-1,])
    doc.df <- data.frame(t(sapply(as.character(doc.df[,1]), function(y) strsplit(y,split=":")[[1]])), stringsAsFactors = F)
    doc.df <- data.frame(data.matrix(doc.df))
    document <- data.frame(matrix(i, nrow(doc.df), 1))
    doc.df <- cbind(document, doc.df)
    colnames(doc.df) <- c("DocID","WordIndex","TF_IDF")
    data.df <- rbind(data.df, doc.df)
  }
  maxWordIndex <- max(data.df$WordIndex)
  tdm = sparseMatrix(i = data.df$DocID, j = data.df$WordIndex, x = data.df$TF_IDF)
  return(list(tdm = tdm,y = y))
}

# Create the output matrix
outputMatrix <- function(y)
{
  y_output <- sparseMatrix(i = y$Category, j = y$DocID, x = 1)
}

# implement logistic regression
# Step 1: Cost Function
lrCostFunction <- function(theta, X, y, lambda, m)
{
  # implement the cost function
  cost = 0
  cost = (sum(0.5*lambda*theta[-1]^2) - (t(y-1)%*%(X%*%theta) + sum(log(sigmoidFN(X%*%theta)))))/m
  reg_term = theta
  reg_term[1] = 0
  grad = (lambda*reg_term - (t(X)%*%(y - sigmoidFN(X%*%theta))))/m
  return(list(cost = cost, grad = as.vector(grad)))
}

# Step 2: The minimizer function
minimizer <- function(max_iter, theta, X, y, lambda, alpha, m)
{
  for(i in 1:max_iter)
  {
    result <- lrCostFunction(theta, X, y,lambda, m)
    theta <- theta - alpha*result$grad
  }
  return(theta)
}

# Step 3: Train the logistic regression model
trainLR <- function(X, y, lambda)
{
  m = nrow(X)
  alpha <- 0.3
  theta <- Matrix(0, nrow(y), ncol(X), sparse = T)
  max_iter <- 2
  for(i in 1:nrow(theta))
  {
    theta[i,] <- minimizer(max_iter,theta[i,], X, y[i,], lambda, alpha, m)
  }
  return(theta)
}

# Predict Ratings
predictLR <- function(Xtest, theta, ytest)
{
  result <- sigmoidFN(Xtest%*%t(theta))
  predLabels <- apply(result, 1, function(x){which.max(x)})
  predLabels <- data.frame(predLabels)
  Output <- cbind(predLabels, ytest)
  colnames(Output) <- c("Pred_Class", "True_Class")
  return(Output)
}

# Sigmoid function
sigmoidFN <- function(x)
{
  result <- 1.0/(1.0 + exp(-x))
  return(result)
}

# Adding bias function
bias <- function(X)
{
  biasVect <- Matrix(1, nrow(X), 1, sparse = T)
  X <- cBind(biasVect, X)
  return(X)
}

run.sh <- function(filename = "DATA.txt")
{
  initPar <- readData(filename)
  trainPath <- data.frame(unlist(strsplit(initPar[1,1], "=")), stringsAsFactors = F)[2,1]
  testPath <- data.frame(unlist(strsplit(initPar[2,1], "=")), stringsAsFactors = F)[2,1]
  temp <- data.frame(unlist(strsplit(initPar[3,1], "=")), stringsAsFactors = F)[2,1]
  C <- data.frame(unlist(strsplit(temp, ",")), stringsAsFactors = F)
  colnames(C) <- c("C")
  trainingData <- readData(trainPath)
  testingData <- readData(testPath)
  trainingData <- TermDocMatrix(trainingData)
  testingData <- TermDocMatrix(testingData)
  X <- trainingData$tdm
  X = bias(X)
  
  y <- outputMatrix(trainingData$y)
  
  Xtest <- testingData$tdm
  Xtest <- bias(Xtest)
  Xtest <- Xtest[,1:ncol(X)]
  ytest <- testingData$y$Category
  for(i in 1:nrow(C))
  {
    lambda <- as.numeric(C[i,1])
    theta <- trainLR(X, y, lambda)
    result <- predictLR(Xtest, theta, ytest)
    outputfile <- paste0("Output_Using_C=",lambda)
    write.table(result, file = outputfile, sep = " ", row.names = F, col.names = F, quote = F)
  }
}

