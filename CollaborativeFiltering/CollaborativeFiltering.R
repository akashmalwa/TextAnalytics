#install.packages("Matrix")
library(Matrix)

# Read training data from train.csv and store it in a data frame
readTrainData = function(filename)
{
  trainingData = read.csv(filename, header = F)
  colnames(trainingData) = c("MovieID","UserID","Rating","RatingDate")
  trainingData = trainingData[,!(names(trainingData) %in% c("RatingDate"))]
  # increment the user ID and movie ID by 1 to match with the indexing standard in R
  trainingData$MovieID = trainingData$MovieID + 1
  trainingData$UserID = trainingData$UserID + 1  
  return(trainingData)
}

# Read the test set and development set in a data frame
readTestData = function(filename)
{
  testData = read.csv(filename, header = F)
  colnames(testData) = c("MovieID", "UserID")
  # increment the user ID and movie ID by 1 to match with the indexing standard in R
  testData$MovieID = testData$MovieID + 1
  testData$UserID = testData$UserID + 1
  return(testData)
}

# Build the User Movie matrix from the training data
buildUserMovieMatrix = function(trainingData)
{
  UIMatrix = sparseMatrix(i = trainingData$UserID, j = trainingData$MovieID, x = trainingData$Rating)
  return(UIMatrix)
}

# Starting point of the code
recommenderSystem = function()
{
  # read data from the CSV files
  # read training data
  trainCSV = readline(prompt = "Enter the name of the training data CSV: ")
  trainingData = readTrainData(trainCSV)
  testCSV = readline(prompt = "Enter the name of the testing CSV: ")
  testingData = readTestData(testCSV)
  # build the user item sparse matrix
  UIMatrix = buildUserMovieMatrix(trainingData)
  # corpus exploration of the training data
  cat("CORPUS EXPLORATION \n")
  corpusExploration(trainingData, UIMatrix)
  readline("EXECUTION PAUSED. PRESS ENTER TO CONTINUE WITH THE EXPERIMENTS")
  Experiment1(UIMatrix, testingData)
  readline("EXECUTION PAUSED. PRESS ENTER TO CONTINUE WITH THE EXPERIMENTS")
  Experiment2(UIMatrix, testingData)
  readline("EXECUTION PAUSED. PRESS ENTER TO CONTINUE WITH THE EXPERIMENTS")
  Experiment3(UIMatrix, testingData)
  readline("EXECUTION PAUSED. PRESS ENTER TO CONTINUE WITH THE EXPERIMENTS")
  #   Experiment4(UIMatrix, testingData)
  cat("Experiments complete, results written to the respective files \n")
  cat("APR_CS stands for Average predicted Rating using Cosine Similarity \n")
  cat("WAPR_CS stands for Weighted Average Predicted Rating using Cosine Similarity \n")
  cat("APR_DP stands for Average Predicted Rating using Dot Product \n")
  return(UIMatrix)
}

# Experiment 1
Experiment1 = function(UIMatrix, testingData)
{
  # step 1: impute matrix for experiment 1 using option 2
  cat("Starting Experiment #1 \n")
  k = readline(prompt = "Enter the value of neighbours for the experiment: ")
  k = as.numeric(k)
  result = matrixImputation(UIMatrix)
  userRatingMatrix = result$mat
  meanVect = result$meanVect
  flag = F
  # find norms of each vector in the matrix
  userNorms = findVectorNorms(userRatingMatrix)
  predictionScoreCS = data.frame(matrix(0, nrow = nrow(testingData),2))
  colnames(predictionScoreCS) = c("APR", "WAPR")
  predictionScoreDP = data.frame(matrix(0, nrow = nrow(testingData),1))
  colnames(predictionScoreDP) = c("APR")
  # set the progress bar
  timer = proc.time()
  progressBar = txtProgressBar(min = 0, max = nrow(testingData), style = 3)
  for(i in 1:nrow(testingData))
  {
    setTxtProgressBar(progressBar, i)
    predictedScores = memoryBasedCF_Exp1(testingData$MovieID[i], testingData$UserID[i], userRatingMatrix, userRatingMatrix, meanVect, userNorms, k, flag)
    predictionScoreCS$APR[i] = predictedScores$APR_CS
    predictionScoreCS$WAPR[i] = predictedScores$WAPR_CS
    predictionScoreDP$APR[i] = predictedScores$APR_DP
  }
  cat("\n")
  cat(proc.time() - timer)
  cat("\n")
  # write the prediction scores from the first experiment to a file
  write.table(predictionScoreCS$APR, file = "Experiment-1-APR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using cosine similarity written to Experiment-1-APR-CS.txt \n")
  write.table(predictionScoreCS$WAPR, file = "Experiment-1-WAPR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Weighted average predicted ratings using cosine similarity written to Experiment-1-WAPR-CS.txt \n")
  write.table(predictionScoreDP$APR, file = "Experiment-1-APR-DP.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using dot product written to Experiment-1-APR-DP.txt \n")
}

# Experiment 2
Experiment2 = function(UIMatrix, testingData)
{
  cat("Starting Experiment #2 \n")
  k = readline(prompt = "Enter the value of neighbours for the experiment: ")
  k = as.numeric(k)
  # compute matrix A using dot product similarity
  result = matrixImputation(UIMatrix)
  UIMatrix = result$mat
  UIMatrix_DPS = computeA_DPS(UIMatrix)
  # compute matrix A using cosine similarity
  UIMatrix_CS = computeA_CS(UIMatrix)
  predictionScoreCS = data.frame(matrix(0, nrow = nrow(testingData),2))
  colnames(predictionScoreCS) = c("APR", "WAPR")
  predictionScoreDP = data.frame(matrix(0, nrow = nrow(testingData),1))
  colnames(predictionScoreDP) = c("APR")
  timer = proc.time()
  progressBar = txtProgressBar(min = 0, max = nrow(testingData), style = 3)
  for(i in 1:nrow(testingData))
  {
    setTxtProgressBar(progressBar, i)
    predictedScores = modelBasedCF_Exp2(testingData$MovieID[i], testingData$UserID[i], UIMatrix_DPS, UIMatrix_CS, UIMatrix, k)
    predictionScoreCS$APR[i] = predictedScores$APR_CS
    predictionScoreCS$WAPR[i] = predictedScores$WAPR_CS
    predictionScoreDP$APR[i] = predictedScores$APR_DP
  }
  cat("\n")
  cat(proc.time() - timer)
  cat("\n")
  write.table(predictionScoreCS$APR, file = "Experiment-2-APR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using cosine similarity written to Experiment-2-APR-CS.txt \n")
  write.table(predictionScoreCS$WAPR, file = "Experiment-2-WAPR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Weighted average predicted ratings using cosine similarity written to Experiment-2-WAPR-CS.txt \n")
  write.table(predictionScoreDP$APR, file = "Experiment-2-APR-DP.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using dot product written to Experiment-2-APR-DP.txt \n")
}

# Experiment 3 - Part 1
Experiment3 = function(UIMatrix, testingData)
{
  cat("Starting Experiment #3 \n")
  k = readline(prompt = "Enter the value of neighbours for the experiment: ")
  k = as.numeric(k)
  result = matrixCentralization(UIMatrix)
  userRatingMatrix = result$mat
  meanVect = result$meanVect
  normalizedMatrix = standardizeMatrix(userRatingMatrix)
  userNorms = findVectorNorms(normalizedMatrix)
  flag = FALSE
  predictionScoreCS = data.frame(matrix(0, nrow = nrow(testingData),2))
  colnames(predictionScoreCS) = c("APR", "WAPR")
  predictionScoreDP = data.frame(matrix(0, nrow = nrow(testingData),1))
  colnames(predictionScoreDP) = c("APR")
  # set the progress bar
  timer = proc.time()
  progressBar = txtProgressBar(min = 0, max = nrow(testingData), style = 3)
  for(i in 1:nrow(testingData))
  {
    setTxtProgressBar(progressBar, i)
    predictedScores = memoryBasedCF_Exp1(testingData$MovieID[i], testingData$UserID[i], userRatingMatrix, normalizedMatrix, meanVect, userNorms, k, flag)
    predictionScoreCS$APR[i] = predictedScores$APR_CS
    predictionScoreCS$WAPR[i] = predictedScores$WAPR_CS
    predictionScoreDP$APR[i] = predictedScores$APR_DP
  }
  # write the prediction scores from the first experiment to a file
  cat("\n")
  cat(proc.time() - timer)
  cat("\n")
  write.table(predictionScoreCS$APR, file = "Experiment-3-Part1-APR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using cosine similarity written to Experiment-3-Part1-APR-CS.txt \n")
  write.table(predictionScoreCS$WAPR, file = "Experiment-3-Part1-WAPR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Weighted average predicted ratings using cosine similarity written to Experiment-3-Part1-WAPR-CS.txt \n")
  write.table(predictionScoreDP$APR, file = "Experiment-3-Part1-APR-DP.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using dot product written to Experiment-3-Part1-APR-DP.txt \n")
}


# Experiment 1 - user user similarity
memoryBasedCF_Exp1 = function(movieID, userID, userRatingMatrix, normalizedMatrix, meanVect, userNorms, k, flag)
{
  # if experiment 4 is being executed take the dot product with k user clusters
  if(flag)
  {
    dotProduct = userRatingMatrix%*%normalizedMatrix[userID,]
    colnames(dotProduct) = c("DotProduct")
  }
  # calculate the dot product of the query with each of the user vectors
  else
  {
    dotProduct = normalizedMatrix%*%normalizedMatrix[userID,]
    colnames(dotProduct) = c("DotProduct")
  }
  
  # check if the dot product is zero i.e. if the user does not occur in the training data
  if(length(which(dotProduct != 0)) == 0)
  {
    if(length(which(normalizedMatrix[,movieID] != 0)) == 0)
    {
      APR_CS = meanVect[userID]
      WAPR_CS = APR_CS
      APR_DP = APR_CS
    }
    else
    {
      APR_CS = sum(normalizedMatrix[,movieID])/length(which(normalizedMatrix[,movieID] != 0)) + 3
      WAPR_CS = APR_CS
      APR_DP = APR_CS
    }
  }
  else
  {
    # compute the cosine similarity between the users
    cosineSimilarity = Matrix(0, nrow = nrow(dotProduct), 1, sparse = T)
    colnames(cosineSimilarity) = c("CosineSimilarity")
    cosineSimilarity[which(!dotProduct == 0), 1] = dotProduct[which(!dotProduct == 0),1]/userNorms[which(!dotProduct == 0)]
    if(flag)
      cosineSimilarity = cosineSimilarity/norm(as.matrix(normalizedMatrix[userID,]), "F")
    else
      cosineSimilarity = cosineSimilarity/userNorms[userID]
    # find indices of the k nearest neighbors using cosine similarity and dot product
    KNN_CS = apply(cosineSimilarity, 2, order, decreasing = T)[1:k, ]
    KNN_DP = apply(dotProduct, 2, order, decreasing = T)[1:k,]
    # remove the queried user from being considered its own neighbor
    KNN_CS = KNN_CS[!KNN_CS %in% userID]
    KNN_DP = KNN_DP[!KNN_DP %in% userID]
    
    # compute the predicted score for the given userID and movieID
    aggregatedAPR_CS = sum(userRatingMatrix[KNN_CS, movieID])
    aggregatedAPR_DP = sum(userRatingMatrix[KNN_DP, movieID])
    APR_CS = (aggregatedAPR_CS/k) + meanVect[userID]
    APR_DP = (aggregatedAPR_DP/k) + meanVect[userID]
    WAPR_CS = (t(cosineSimilarity[KNN_CS])%*%userRatingMatrix[KNN_CS, movieID])/sum(abs(cosineSimilarity[KNN_CS])) + meanVect[userID]
  }
  predictedScore = list(APR_CS = APR_CS, APR_DP = APR_DP, WAPR_CS = WAPR_CS)
  return(predictedScore)
}

# Model based collaborative filtering
modelBasedCF_Exp2 = function(movieID, userID, UIMatrix_DPS, UIMatrix_CS, UIMatrix, k)
{
  # find movie vector using dot product similarity
  movieVectorDPS = UIMatrix_DPS[movieID,]
  # find knn neighbors items based on dot product similarity
  KNN_DPS = order(movieVectorDPS, decreasing = T)[1:k]
  # remove the queried movie from being considered its own neighbor
  KNN_DPS = KNN_DPS[!KNN_DPS %in% movieID]
  # find the ratings given by the user to the knn items
  ratingsDPS = UIMatrix[userID, KNN_DPS]
  APR_DP = sum(ratingsDPS)/k + 3
  # find knn neighbors items based on the cosine similarity
  movieVectorCS = UIMatrix_CS[movieID,]
  KNN_CS = order(movieVectorCS, decreasing = T)[ 1:k]
  # remove the queried movie from being considered its own neighbor
  KNN_CS = KNN_CS[!KNN_CS %in% movieID]
  # find the ratings given by the user to the knn of the given movieID
  ratingsCS = UIMatrix[userID, KNN_CS]
  # find the mean rating given by the user to the item
  APR_CS = sum(ratingsCS)/k + 3
  weightVector = UIMatrix_CS[movieID, KNN_CS]
  if(sum(abs(weightVector)) != 0)
  {
    weightVector = weightVector/sum(abs(weightVector))
    WAPR_CS = t(ratingsCS)%*%weightVector + 3
  }
  else
  {
    WAPR_CS = APR_CS
  }
  predictedScore = list(APR_CS = APR_CS, APR_DP = APR_DP, WAPR_CS = WAPR_CS)
  return(predictedScore)
}

# Experiment 2
# Compute A using the dot product similarity
computeA_DPS = function(UIMatrix)
{
  A_DPS = t(UIMatrix)%*%UIMatrix
  return(A_DPS)
}

# Compute A using the cosine similarity
computeA_CS = function(UIMatrix)
{
  # normalize the entire matrix before computing the cosine similarity matrix A
  movieNorms = findVectorNorms(t(UIMatrix))
  # transform all the item vectors to the unit vectors
  UIMatrix[,which(movieNorms != 0)] = t(t(UIMatrix[,which(movieNorms != 0)])/movieNorms[which(movieNorms != 0)])
  A_CS = t(UIMatrix)%*%UIMatrix
  return(A_CS)
}

# Experiment 3
# Step 1: Standardize the  user matrix
# Step 2: Call the memory based CF for users
# Step 3: Standardize the item matrix
# Step 4: Call the model based CF for items
# Note: For the above steps I have reused my code for experiments 1 and 2.

# Matrix centralization
matrixCentralization = function(mat)
{
  # center each vector around its mean
  meanVect = apply(mat,1,function(x){mean(x[which(!x == 0)])})
  #   mat = (mat - meanVect)*(mat != 0)
  mat@x = mat@x - meanVect[mat@i+1]
  # find the normalization factors for all the row vectors in the matrix
  result = list(meanVect = meanVect, mat = mat)
  return(result)
}

# Matrix standardization 
standardizeMatrix = function(mat)
{
  vectNorms = findVectorNorms(mat)
  mat[which(vectNorms != 0),] = mat[which(vectNorms != 0),]/vectNorms[which(vectNorms != 0)]
  return(mat)
}

# Matrix imputation
matrixImputation = function(mat)
{
  mat[which(!mat == 0)] = mat[which(!mat == 0)] - 3
  meanVect = matrix(3, nrow = nrow(mat), ncol = 1)
  result = list(mat = mat, meanVect = meanVect)
  return(result)
}

# Experiment 4
# Bipartite clustering implementation
# Clustering on users
Experiment4 = function(UIMatrix, testingData)
{
  cat("Starting Experiment #4 Part 1\n")
  cat("Clustering done on users \n")
  k1 = readline(prompt = "Enter number of movie clusters: ")
  k1 = as.numeric(k1)
  k2 = readline(prompt = "Enter number of users clusters: ")
  k2 = as.numeric(k2)
  UIMatrixCopy = UIMatrix
  # remove users that do not exist in our training data
  UIMatrix = UIMatrix[apply(UIMatrix!=0, 1, any), , drop=FALSE]
  # perform bipartite clustering on users
  timer = proc.time()
  userCentroidMatrix = bipartite_clustering(UIMatrix, k1, k2)
  cat("Time spent in Bipartite clustering = ", proc.time() - timer, "\n")
  # standardize the original user item matrix
  UIMatrix = UIMatrixCopy
  # perform matrix centralization on the user matrix
  result = matrixCentralization(UIMatrix)
  UIMatrix = result$mat
  meanVect = result$meanVect
  # perform normalization for the user centroids
  result <- matrixCentralization(userCentroidMatrix)
  userCentroidMatrix <- result$mat
  userCentroidMatrix <- standardizeMatrix(userCentroidMatrix)
  # calculate the centroid norms
  centroidNorms = findVectorNorms(userCentroidMatrix)
  flag = T
  k = readline(prompt = "Enter the value of neighbours for the experiment: ")
  k = as.numeric(k)
  predictionScoreCS = data.frame(matrix(0, nrow = nrow(testingData),2))
  colnames(predictionScoreCS) = c("APR", "WAPR")
  predictionScoreDP = data.frame(matrix(0, nrow = nrow(testingData),1))
  colnames(predictionScoreDP) = c("APR")
  timer = proc.time()
  progressBar = txtProgressBar(min = 0, max = nrow(testingData), style = 3)
  for(i in 1:nrow(testingData))
  {
    setTxtProgressBar(progressBar, i)
    predictedScores = memoryBasedCF_Exp1(testingData$MovieID[i], testingData$UserID[i], userCentroidMatrix, UIMatrix, meanVect, centroidNorms, k, flag)
    predictionScoreCS$APR[i] = predictedScores$APR_CS
    predictionScoreCS$WAPR[i] = predictedScores$WAPR_CS
    predictionScoreDP$APR[i] = predictedScores$APR_DP
  }
  cat("\n")
  cat(proc.time() - timer)
  cat("\n")
  write.table(predictionScoreCS$APR, file = "Experiment-4-Part1-APR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using cosine similarity written to Experiment-3-Part1-APR-CS.txt \n")
  write.table(predictionScoreCS$WAPR, file = "Experiment-4-Part1-WAPR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Weighted average predicted ratings using cosine similarity written to Experiment-3-Part1-WAPR-CS.txt \n")
  write.table(predictionScoreDP$APR, file = "Experiment-4-Part1-APR-DP.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using dot product written to Experiment-3-Part1-APR-DP.txt \n")
  
  # movie - movie similarity
  UIMatrix = UIMatrixCopy
  result = matrixImputation(UIMatrix)
  UIMatrix = result$mat
  cat("Starting Experiment 4 Part #2")
  UIMatrix_DPS = computeA_DPS(userCentroidMatrix)
  # compute matrix A using cosine similarity
  UIMatrix_CS = computeA_CS(userCentroidMatrix)
  
  predictionScoreCS = data.frame(matrix(0, nrow = nrow(testingData),2))
  colnames(predictionScoreCS) = c("APR", "WAPR")
  predictionScoreDP = data.frame(matrix(0, nrow = nrow(testingData),1))
  colnames(predictionScoreDP) = c("APR")
  timer = proc.time()
  progressBar = txtProgressBar(min = 0, max = nrow(testingData), style = 3)
  for(i in 1:nrow(testingData))
  {
    setTxtProgressBar(progressBar, i)
    predictedScores = modelBasedCF_Exp2(testingData$MovieID[i], testingData$UserID[i], UIMatrix_DPS, UIMatrix_CS, UIMatrix, k)
    predictionScoreCS$APR[i] = predictedScores$APR_CS
    predictionScoreCS$WAPR[i] = predictedScores$WAPR_CS
    predictionScoreDP$APR[i] = predictedScores$APR_DP
  }
  cat("\n")
  cat(proc.time() - timer)
  cat("\n")
  write.table(predictionScoreCS$APR, file = "Experiment-4-Part2-APR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using cosine similarity written to Experiment-4-Part2-APR-CS.txt \n")
  write.table(predictionScoreCS$WAPR, file = "Experiment-4-Part2-WAPR-CS.txt", sep = "\n", row.names = F, col.names = F)
  cat("Weighted average predicted ratings using cosine similarity written to Experiment-4-Part2-WAPR-CS.txt \n")
  write.table(predictionScoreDP$APR, file = "Experiment-4-Part2-APR-DP.txt", sep = "\n", row.names = F, col.names = F)
  cat("Average predicted ratings using dot product written to Experiment-4-Part2-APR-DP.txt \n")
}


bipartite_clustering <- function(TDM, k1, k2)
{
  numOfIterations = 1
  TDM_Original = TDM
  flag = TRUE
  while(numOfIterations <= 5)
  {
    # cluster words in the data
    cat(paste("BIPARTITE CLUSTERING ITERATION =",numOfIterations), "\n")
    vectAssignments = k_means(t(TDM), k1)
    newTDM = aggregateDocWords(t(TDM_Original), vectAssignments$uniqueCosineSimilarity, k1, flag)
    newTDM = t(newTDM)
    flag = FALSE
    # store the sum of cosine similarity for words
    sumWordCosineSimilarity = sum(vectAssignments$uniqueCosineSimilarity)
    # cluster documents in the data
    vectAssignments = k_means(newTDM, k2)
    # aggregate document vectors based on cluster assignments
    
    newTDM = aggregateDocWords(TDM_Original, vectAssignments$uniqueCosineSimilarity, k2, flag)
    TDM = newTDM
    flag = TRUE
    sumDocCosineSimilarity = sum(vectAssignments$uniqueCosineSimilarity)
    numOfIterations = numOfIterations+1
  }
  # create a file to write the internal evaluation metrics
  write.table(data.frame(matrix(c("Word Cluster Size", "Doc Cluster Size", "Sum of cosine similarity for words", "Sum of cosine similarity for documents"), ncol = 4, nrow = 1)), 
              file = "Internal Evaluation Metrics.csv", sep = ",", append = T, col.names = F, row.names = F)
  # write internal evaluation metrics to a file
  internalMetrics = data.frame(matrix(c(k1, k2, sumWordCosineSimilarity, sumDocCosineSimilarity), 1, 4, byrow = T))
  write.table(internalMetrics, file = "Internal Evaluation Metrics.csv", sep = ",", append = T, row.names = F, col.names = F)
  return(TDM)
}

# Clustering with k-means algorithm
k_means <- function(tdm, k)
{
  vectorNorms <- findVectorNorms(tdm)
  vectorNorms[which(vectorNorms == 0)] = 1
  # initialize cluster centroids initially as k randomly selected seeds
  clusterCentroids <- tdm[sample(nrow(tdm), size = k, replace = FALSE),]
  stopping_criteria_not_met = TRUE
  i = 1
  # while stopping criteria is not met
  while(stopping_criteria_not_met)
  {
    clusterNorms = findVectorNorms(clusterCentroids)
    clusterNorms[which(clusterNorms == 0)] = 1
    # find closest centroids to the data points
    vectAssignments = findClosestCentroids(tdm, clusterCentroids, vectorNorms, clusterNorms)
    # recompute centroids for the new cluster
    newClusterCentroids = recomputingCentroids(clusterCentroids, vectAssignments$uniqueIndexMatrix, tdm)
    # find whether the centers move or not
    clusterDifference = find_K_Means_Convergence(clusterCentroids, newClusterCentroids)
    clusterCentroids = newClusterCentroids
    i = i + 1
    stopping_criteria_not_met = (i <= 10 && (clusterDifference > 0.1))
  }
  return(vectAssignments)
}

# Find the vector norms of the term document matrix
findVectorNorms <- function(mat)
{
  vectorNorms <- vector(mode = "numeric",nrow(mat))
  vectorNorms <- apply(mat, 1, function(x) norm(as.matrix(x),"F"))
  return(vectorNorms)
}

# Find closest centroids for the data points using cosine similarity
# VectorNorms contain the norms of all of the document vectors
# clusterNorms contain the norms of all of the cluster centroids
findClosestCentroids <- function(tdm, clusterCentroids, vectorNorms, clusterNorms)
{
  cat("Finding closest centroids to the document or word vectors \n")
  cosineSimilarityMatrix = tdm %*% t(clusterCentroids)
  cosineSimilarityMatrix = cosineSimilarityMatrix / vectorNorms
  cosineSimilarityMatrix = t(t(cosineSimilarityMatrix) / clusterNorms)
  
  # break ties between the cluster assignments
  uniqueIndexMatrix <- Matrix(0, nrow(cosineSimilarityMatrix), ncol(cosineSimilarityMatrix), sparse = T)
  
  uniqueIndexMatrix[cbind(seq_len(nrow(cosineSimilarityMatrix)),max.col(cosineSimilarityMatrix, "random"))] <- 1
  uniqueCosineSimilarity = cosineSimilarityMatrix * uniqueIndexMatrix
  return(list(uniqueIndexMatrix = uniqueIndexMatrix, uniqueCosineSimilarity = uniqueCosineSimilarity))
}

# Recompute the centroids of the cluster
recomputingCentroids <- function(clusterCentroids, uniqueIndexVector, tdm)
{
  cat("Recomputing Centroids \n")
  k = nrow(clusterCentroids)
  features = ncol(clusterCentroids)
  newClusterCentroids <- Matrix(0, nrow = k, ncol = features, sparse = T)
  for(i in 1:k)
  {
    indexVector = uniqueIndexVector[,i]
    clusterDocuments = tdm * indexVector
    numOfDocAssigned = sum(indexVector)
    if(numOfDocAssigned > 0)
    {
      newClusterCentroids[i,] = colSums(clusterDocuments)/numOfDocAssigned
    }
    else
    {
      newClusterCentroids[i,] = clusterCentroids[i,]
    }
  }
  return(newClusterCentroids)
}


# Find if the centroids are still moving or not
find_K_Means_Convergence <- function(clusterCentroids, newClusterCentroids)
{
  cat("Finding centroid difference between the new cluster and the old cluster \n")
  clusterDifference = clusterCentroids - newClusterCentroids
  clusterNorm = norm(clusterDifference)
  
  cat(paste("Difference between the old and the new cluster centroids =",clusterNorm), "\n")
  return(clusterNorm)
}

# Aggregate Words belonging to the same cluster
aggregateDocWords <- function(tdm, uniqueCosineSimilarity, k, flag)
{
  if(flag)
    cat("Aggregating words \n")
  else
    cat("Aggregating documents \n")
  newTDM <- Matrix(0, nrow = k, ncol = ncol(tdm), sparse = T)
  for(i in 1:k)
  {
    cosineSimilarityVect = uniqueCosineSimilarity[,i]
    aggregateVect = colSums(tdm * cosineSimilarityVect)
    if(sum(aggregateVect) > 0)
    {
      newTDM[i,] = aggregateVect  
    }
  }
  cat("Aggregation complete \n")
  return(newTDM)
}


# Find norms of the vectors (Users or items) in the UI matrix
findVectorNorms = function(mat)
{
  vectorNorms = vector(mode = "numeric",nrow(mat))
  vectorNorms = apply(mat, 1, function(x) norm(as.matrix(x),"F"))
  return(vectorNorms)
}

# Corpus Exploration
corpusExploration = function(trainingData, UIMatrix)
{
  cat(paste0("Total number of movies in the training set = ", length(unique(trainingData$MovieID)), "\n"))
  cat(paste0("Total number of users in the training set = ", length(unique(trainingData$UserID)), "\n"))
  cat(paste0("Number of times any movie was rated 1 = ", length(which(trainingData$Rating == 1)), "\n"))
  cat(paste0("Number of times any movie was rated 3 = ", length(which(trainingData$Rating == 3)), "\n"))
  cat(paste0("Number of times any movie was rated 5 = ", length(which(trainingData$Rating == 5)), "\n"))
  cat(paste0("The average movie rating across all users and movies =",sum(UIMatrix)/length(which(UIMatrix != 0)), "\n"))
  cat(paste0("Number of movies rated by user 4321 = ", length(which(UIMatrix[4322,] != 0)), "\n"))
  cat(paste0("Number of times user 4321 gave a rating of 1 to a movie = ", length(which(UIMatrix[4322,] == 1)), "\n"))
  cat(paste0("Number of times user 4321 gave a rating of 3 to a movie = ", length(which(UIMatrix[4322,] == 3)), "\n"))
  cat(paste0("Number of times user 4321 gave a rating of 5 to a movie = ", length(which(UIMatrix[4322,] == 5)), "\n"))
  cat(paste0("Average rating given by user 4321 = ", sum(UIMatrix[4322,])/length(which(UIMatrix[4322,] != 0)), "\n"))
  cat(paste0("Number of users rating movie 3 = ", length(which(UIMatrix[,4] != 0)),"\n"))
  cat(paste0("Number of times movie 3 was rated 1 = ", length(which(UIMatrix[,4] == 1)),"\n"))
  cat(paste0("Number of times movie 3 was rated 3 = ", length(which(UIMatrix[,4] == 3)),"\n"))
  cat(paste0("Number of times movie 3 was rated 5 = ", length(which(UIMatrix[,4] == 5)),"\n"))
  cat(paste0("Average rating for movie 3 = ", sum(UIMatrix[,4])/length(which(UIMatrix[,4] != 0)),"\n"))
}
