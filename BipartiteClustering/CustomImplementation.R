# Uncomment the next line to install the required packages that I used for running my code
# install.packages("Matrix")

library("Matrix")

# DEVELOPMENT DATASETS
#docVectors = "HW2_dev.docVectors"
#wordDict = "HW2_dev.dict"
#docFreq = "HW2_dev.df"

# TESTING DATASETS
#docVectors = "HW2_test.docVectors"
#wordDict = "HW2_test.dict"
#docFreq = "HW2_test.df"

# Function creates the word dictionary from a dict file
# Takes argument the name of the dict file
createWordDict <- function(filename)
{
  wordDict.table = read.table(filename, sep = " ")
  wordDict.df = data.frame(wordDict.table, stringsAsFactors = F)
  wordDict.df <- na.omit(wordDict.df)
  colnames(wordDict.df) <- c("Word", "TermIndex")
  wordDict.df = data.frame(wordDict.df[order(wordDict.df$TermIndex),])
  wordDict.df = incrementTermIndexes(wordDict.df)
  return(wordDict.df)
}

# Function creates the document frequency data frame used to calculate the inverse document frequency
# Takes two arguments: the name of the file containing the document frequency and the number of documents in the corpus
createDocFreqDF <- function(filename, N)
{
  docFreq.table = read.table(filename, sep = ":")
  docFreq.df = data.frame(docFreq.table)
  docFreq.df <- na.omit(docFreq.df)
  # Convert all the values in the data frame to numeric
  docFreq.df <- data.frame(data.matrix(docFreq.df))
  # Calculating the inverse document frequency from the document frequency
  idf = lapply(docFreq.df[,2], function(x) x = log(N/(x+1)))
  idf = data.frame(unlist(idf))
  docFreq.df <- cbind(docFreq.df, idf)
  colnames(docFreq.df) <- c("TermIndex", " DocumentFrequency", "InverseDocumentFrequency")
  docFreq.df <- incrementTermIndexes(docFreq.df)
  return(docFreq.df)
}

# Create a sparse term document matrix with all entries set to zero
createSparseMatrix <- function(docVect.df)
{
  # N = number of documents
  N = nrow(docVect.df)
  # Create Word Dictionary
  wordDict <- readline(prompt = "Enter the file name of the Word dictionary file: (e.g. HW2_dev.dict) ")
  wordDict.df <- createWordDict(wordDict)
  # Calculate document frequency and inverse document frequency data frame
  docFreq <- readline(prompt = "Enter the filename containing the document frequency: (e.g. HW2_dev.df) ")
  docFreq.df <- createDocFreqDF(docFreq, N)
  tdm <- Matrix(0, nrow(docVect.df), nrow(wordDict.df), sparse = TRUE)
  # Setting all the non-zero entries for the term document matrix as function of tf*idf
  
  for(i in 1:nrow(docVect.df))
  {
    doc <- docVect.df[i,1]
    doc.df <- data.frame(unlist(strsplit(doc, " ")), stringsAsFactors = F)
    doc.df <- data.frame(t(sapply(doc.df[,1], function(y) strsplit(y,split=":")[[1]])), stringsAsFactors = F)
    doc.df <- data.frame(data.matrix(doc.df))
    
    colnames(doc.df) <- c("TermIndex", "TermFrequency")
    #doc.df <- incrementTermIndexes(doc.df)
    for(j in 1:nrow(doc.df))
    {
      k = doc.df$TermIndex[j] + 1
      tdm[i,k] = doc.df$TermFrequency[j] * docFreq.df$InverseDocumentFrequency[k]
      tdm[i,k] = doc.df$TermFrequency[j]
    }
  }
  return(tdm)
}

incrementTermIndexes <- function(df)
{
  # Increment all the term indexes by 1 to match the indexing in R
  termIndex <- data.frame(sapply(df$TermIndex, function(y) y = as.numeric(as.numeric(as.character(y)) + 1)[[1]]), stringsAsFactors = F)
  colnames(termIndex) <- c("TermIndex")
  df$TermIndex <- NULL
  df = cbind(df,TermIndex = termIndex)
  return(df)
}

# Takes input the filename of the file containing the document vectors, must be supplied with quotes e.g. "HW2_dev.docVectors"
bipartite_clustering <- function(filenameDocVectors, k1, k2)
{
  conn = file(filenameDocVectors, open = "r")
  docVect = readLines(conn)
  doc_word_vectors = data.frame(docVect, stringsAsFactors = FALSE)
  close(conn)
  TDM <- createSparseMatrix(doc_word_vectors)
  numOfIterations = 1
  TDM_Original = TDM
  flag = TRUE
  while(numOfIterations <= 10)
  {
    # cluster words in the data
    cat(paste("BIPARTITE CLUSTERING ITERATION =",numOfIterations), "\n")
    vectAssignments = k_means(t(TDM), k1)
    newTDM = aggregateDocWords(t(TDM_Original), vectAssignments$uniqueCosineSimilarity, k1, flag)
    newTDM = t(newTDM)
    flag = FALSE
    # store the word cluster assignments in a data frame
    wordToClusterAssignmentMatrix = vectAssignments$uniqueIndexMatrix
    # store the sum of cosine similarity for words
    sumWordCosineSimilarity = sum(vectAssignments$uniqueCosineSimilarity)
    # cluster documents in the data
    vectAssignments = k_means(newTDM, k2)
    # aggregate document vectors based on cluster assignments
    
    newTDM = aggregateDocWords(TDM_Original, vectAssignments$uniqueCosineSimilarity, k2, flag)
    TDM = newTDM
    flag = TRUE
    sumDocCosineSimilarity = sum(vectAssignments$uniqueCosineSimilarity)
    docToClusterAssignmentMatrix = vectAssignments$uniqueIndexMatrix
    numOfIterations = numOfIterations+1
  }
  # create a file to write the internal evaluation metrics
  write.table(data.frame(matrix(c("Word Cluster Size", "Doc Cluster Size", "Sum of cosine similarity for words", "Sum of cosine similarity for documents"), ncol = 4, nrow = 1)), 
              file = "Internal Evaluation Metrics.csv", sep = ",", append = T, col.names = F, row.names = F)
  # write internal evaluation metrics to a file
  internalMetrics = data.frame(matrix(c(k1, k2, sumWordCosineSimilarity, sumDocCosineSimilarity), 1, 4, byrow = T))
  write.table(internalMetrics, file = "Internal Evaluation Metrics.csv", sep = ",", append = T, row.names = F, col.names = F)
  # writing the document assignments to the various clusters to a file
  wordToClusterAssignment = which(wordToClusterAssignmentMatrix != 0, arr.ind = T)
  docToClusterAssignment = which(docToClusterAssignmentMatrix != 0, arr.ind = T)
  docToClusterAssignment[,1] = docToClusterAssignment[,1] - 1
  docToClusterAssignment = data.frame(docToClusterAssignment)
  colnames(docToClusterAssignment) <- c("Document ID", "Cluster ID")
  colnames(wordToClusterAssignment) <- c("Word ID", "Cluster ID")
  write.table(docToClusterAssignment,file = "DocClusteringAssignments.txt",sep = " ",row.names = F,col.names = F)
  write.table(wordToClusterAssignment,file = "WordClusteringAssignments.txt",sep = " ",row.names = F,col.names = F)
  return(docToClusterAssignment)
}
# Clustering with k-means algorithm
k_means <- function(tdm, k)
{
  vectorNorms <- findVectorNorms(tdm)
  # initialize cluster centroids initially as k randomly selected seeds
  clusterCentroids <- tdm[sample(nrow(tdm), size = k, replace = FALSE),]
  stopping_criteria_not_met = TRUE
  i = 1
  # while stopping criteria is not met
  while(stopping_criteria_not_met)
  {
    clusterNorms = findVectorNorms(clusterCentroids)
    # find closest centroids to the data points
    vectAssignments = findClosestCentroids(tdm, clusterCentroids, vectorNorms, clusterNorms)
    # recompute centroids for the new cluster
    newClusterCentroids = recomputingCentroids(clusterCentroids, vectAssignments$uniqueIndexMatrix, tdm)
    # find whether the centers move or not
    clusterDifference = find_K_Means_Convergence(clusterCentroids, newClusterCentroids)
    clusterCentroids = newClusterCentroids
    i = i + 1
    stopping_criteria_not_met = (i < 15 && (clusterDifference > 0.1))
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


