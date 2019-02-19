rm(list = ls())
coocurrences <- function(X,window,lenWalk,nwalk,ncore=detectCores()){
  library(foreach)
  library(doParallel)
  library(text2vec)
  
  X <- X/rowSums(X)
  N <- nrow(X)
  
  cl <- makeCluster(ncore)
  registerDoParallel(cl)
  
  clusterExport(cl, "X", envir = environment())
  clusterExport(cl, "lenWalk", envir = environment())
  
  
  data <- foreach(node = 1:N,
                  .combine = rbind)  %dopar%
                  {
                    bob <- rep(0,lenWalk)
                    for(walk in 1:nwalk){
                      walk <- node
                      curr <- node
                      for (num in 2:lenWalk) {
                        curr <- which(rmultinom(1, 1, X[curr,]) != 0)
                        walk <- c(walk,curr)
                      }
                      bob <- rbind(bob,walk)
                    }
                    bob <- bob[-1,]
                    return(bob)
                  }
  stopImplicitCluster()
  stopCluster(cl)
  
  data <- apply(data,1,function(x){paste(x,collapse  = " ")})
  nrow(data)
  iterator <- itoken(data,tokenizer=space_tokenizer)
  vocabulary <- create_vocabulary(iterator)
  vectorizer <- vocab_vectorizer(vocabulary)
  cooccurrences <- as.matrix(create_tcm(iterator, vectorizer, skip_grams_window=window))
  cooccurrences <- cooccurrences[, order(as.integer(colnames(cooccurrences)))]
  cooccurrences <- cooccurrences[ order(as.integer(rownames(cooccurrences))),]
  return(cooccurrences)
}
buildAdjacency <- function(nodes,edges,weighted = T){
  edges <- as.matrix(edges)
  nodes <- unlist(nodes)
  N <- length(nodes)
  Adj <- matrix(0,nrow = N,ncol = N)
  E <- nrow(edges)
  for(ind in 1:E) {
    i <- which(nodes == edges[ind,1])
    j <- which(nodes == edges[ind,2])
    
    if (weighted) {
      Adj[i,j] <- as.numeric(edges[ind,3])
    }else{
      Adj[i,j] <- 1 
    }
  }
  return(Adj)
}
# Packages installation ####
if (!require(wordVectors)) {
  if (!(require(devtools))) {
    install.packages("devtools")
  }
  devtools::install_github("bmschmidt/wordVectors")
}
if (!require(readr)) {
  install.packages("readr")
}
if (!require(tm)) {
  install.packages("tm")
}
if (!require(foreach)) {
  install.packages("foreach")
}
if (!require(doParallel)) {
  install.packages("doParallel")
}
if (!require(e1071)) {
  install.packages("e1071")
}
classes <- read.table("CITESEER/group.txt", quote="\"")$V2
classes <- as.factor(classes)
#TF-IDF
dT <- read.table("CITESEER/feature.txt",stringsAsFactors = F,header = F,sep = "\t")
N <- nrow(dT)
dT <- dT[,-ncol(dT)]

# View(sort(as.integer(vocab)))
idf <- apply(dT,2,function(d){
  nDoc <- length(d[d > 0])
  return(log(N/nDoc))
})
dTidf <- t(t(dT) * idf)

Bow <- apply(dT,1,function(x){as.character(which(x != 0))})
data <- lapply(Bow,function(x){paste0(x,collapse = " ")})
write.table(data,"CITESEER/tm.txt",quote = F,row.names = F,col.names = F)
prep_word2vec(origin="CITESEER/tm.txt",destination="CITESEER/clean.txt",lowercase=T,bundle_ngrams=1)

graph <- read.delim("../CITESEER/graph.txt", header=FALSE)
graph$weight <- 0
graph <- aggregate(weight ~ V1 + V2, data = graph, FUN = length)
nodes <- sort(unique(c(graph$V1,graph$V2)))
graph <- buildAdjacency(nodes,edges = graph)

#Adj####
X <- graph + t(graph)
X <- X/rowSums(X)

system.time(P <- coocurrences(X,4,80,80))
Neg <- 5 * round((rowSums(P) %*% t(colSums(P))) / sum(P))
Sim <- P - Neg
# Sim <- P
Sim[Sim<0] <- 0
Sim <- Sim/rowSums(Sim)

beta_init <- as.matrix(dT)/rowSums(as.matrix(dT))
C <- Sim %*% beta_init
C <- C / rowSums(C)

unlink("CITESEER/clean_vectors.bin")
model <- train_word2vec("CITESEER/clean.txt","CITESEER/clean_vectors.bin",vectors=200,threads=10,window=wind,iter=20,negative_samples=5,min_count=1)

vocab <- as.integer(rownames(model@.Data)[-1])

vector <- model@.Data[-1,]

vector <- vector[order(as.integer(vocab)),]

EmbeddingDocument <- beta_init %*% as.matrix(vector)
colnames(EmbeddingDocument) <- paste("V",c(1:200),sep="")
EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
write.table(EmbeddingDocument,paste0("CITESEER/Mu0/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)

beta <- (beta_init + C)/(2)
EmbeddingDocument <- beta %*% as.matrix(vector)
colnames(EmbeddingDocument) <- paste("V",c(1:200),sep="")
EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
write.table(EmbeddingDocument,paste0("CITESEER/Mu1/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)

mu_opt <- 1.5
beta <- (beta_init + (mu_opt * C))/(1+mu_opt)
EmbeddingDocument <- beta %*% as.matrix(vector)
colnames(EmbeddingDocument) <- paste("V",c(1:200),sep="")
EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
write.table(EmbeddingDocument,paste0("CITESEER/MuEtoile/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)

accuMatr <- c()
for(num in seq(0.1,0.5,length.out = 5)){
  ntrain <- floor(num * N)
  ntest <- N - ntrain
  accu <- 0
  for (i in 1:10){
    idtrain <- sample(1:N,ntrain)
    idtest <- setdiff(1:N,idtrain)
    
    #tadw
    
    trainData <- EmbeddingDocument[idtrain,]
    testData <- EmbeddingDocument[idtest,]
    model <- svm(trainData,classes[idtrain], probability = TRUE)
    pred_prob <- predict(model, testData, decision.values = TRUE, probability = TRUE)
    accu <- accu + sum(diag(table(pred_prob,classes[idtest])))/ntest
  }
  accuMatr <-c( accuMatr,accu/10)
} 
