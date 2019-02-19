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

classes <- read.table("CORA2/group.txt", quote="\"")
na <- which(is.na(classes))
nodes <- which(!is.na(classes)) -1
classes <- classes[-na,]
classes <- as.factor(classes)

data <- as.vector(read_csv("CORA2/data.txt",
                           col_names = FALSE))

tmCorpus <- Corpus(VectorSource(data$X1))
tmCorpus <- tm_map(tmCorpus, stripWhitespace)
tmCorpus <- tm_map(tmCorpus, content_transformer(tolower))
tmCorpus <- tm_map(tmCorpus, removeNumbers)
tmCorpus <- tm_map(tmCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)

tmCorpus <- tm_map(tmCorpus, removeWords, stopwords("english"))
txt <- data.frame(text = get("content", tmCorpus),stringsAsFactors = FALSE)$text
write.table(txt,"CORA2/tm.txt",quote = F,row.names = F,col.names = F)

prep_word2vec(origin="CORA2/tm.txt",destination="CORA2/clean.txt",lowercase=T,bundle_ngrams=1)
Bow <- lapply(txt,Boost_tokenizer)

graph <- read.delim("CORA2/graph.txt", header=FALSE)
graph$weight <- 0
graph <- aggregate(weight ~ V1 + V2, data = graph, FUN = length)

graph <- buildAdjacency(nodes,edges = graph)

#Adj####
X <- graph + t(graph)
X <- X/rowSums(X)

  unlink("CORA2/clean_vectors2.bin")
  model <- train_word2vec("CORA2/clean.txt","CORA2/clean_vectors2.bin",vectors=200,threads=10,window=15,iter=20,negative_samples=5)
  
  vocab <- rownames(model@.Data)
  
  dT <- matrix(0,nrow = length(Bow), ncol = length(vocab))
  
  for (i in 1:length(Bow)) {
    for (mot in Bow[[i]]) {
      j <- which(vocab == tolower(mot))
      dT[i,j] <- dT[i,j] + 1
    }
  }
  N <- nrow(dT)
  
  idf <- apply(dT,2,function(d){
    nDoc <- length(d[d > 0])
    return(log(N/nDoc))
  })
  dTidf <- t(t(dT) * idf)
  
  dT <- dT[-na,]
  dTidf <- dTidf[-na,]
  N <- nrow(dT)
  
  
  noScope <- which(colSums(dT) == 0)
  
  dT <- dT[,-noScope]
  dTidf <- dTidf[,-noScope]
  vocab <- vocab[-noScope]
  
  vector <- model@.Data[-noScope,]
  
  system.time(P <- coocurrences(X,6,80,80))
  Neg <- 5 * round((rowSums(P) %*% t(colSums(P))) / sum(P))
  Sim <- P - Neg
  Sim[Sim<0] <- 0
  Sim <- Sim/rowSums(Sim)
  
  beta_init <- as.matrix(dT)/rowSums(as.matrix(dT))
  C <- Sim %*% beta_init
  C <- C / rowSums(C)
  
  EmbeddingDocument <- beta_init %*% as.matrix(vector)
  colnames(EmbeddingDocument) <- paste("V",c(1:200),sep="")
  EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
  write.table(EmbeddingDocument,paste0("CORA2/Mu0/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)
  
  beta <- (beta_init + C)/(2)
  EmbeddingDocument <- beta %*% as.matrix(vector)
  EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
  write.table(EmbeddingDocument,paste0("CORA2/Mu1/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)
  
  mu_opt <- 1.5
  beta <- (beta_init + (mu_opt * C))/(1+mu_opt)
  EmbeddingDocument <- beta %*% as.matrix(vector)
  EmbeddingDocument <- t(t(EmbeddingDocument)/unlist(apply(EmbeddingDocument,2,function(x){sqrt(t(x) %*% x)})))
  write.table(EmbeddingDocument,paste0("CORA2/MuEtoile/vector.txt",collapse=""),sep=" ",row.names=F,col.names=F)
