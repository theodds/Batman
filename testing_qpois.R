library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)
library(tidyverse)

P <- 10
N <- 2000
sigma <- 20
num_tree <- 50

rmse <- function(x,y) sqrt(mean((x-y)^2))

sim_fried_pois <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  lambda <- sigma * f(X)
  Y <- 20 * rpois(n = n, lambda = lambda / 20)
  
  return(list(X = X, Y = Y, lambda = lambda))
}

c(X,Y,lambda) %<-% sim_fried_pois(N,P,sigma)
probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)

system.time({
  out <- QPoisBart(X, Y, X_test = X, probs = probs, num_trees = num_tree, scale_lambda = 1, 
                  scale_lambda_0 = 1 / sqrt(num_tree),
                  num_burn = 2000, num_thin = 1, num_save = 2000)  
})

system.time({
  out_pos <- PoisBart(X, Y, X_test = X, probs = probs, num_trees = num_tree, scale_lambda = 1, 
                   scale_lambda_0 = 1 / sqrt(num_tree),
                   num_burn = 2000, num_thin = 1, num_save = 2000)  
})

plot(log(lambda), colMeans(out$lambda))
abline(a=0,b=1)

plot(log(lambda), colMeans(out_pos$lambda))
rmse(log(lambda), colMeans(out_pos$lambda))
rmse(log(lambda), colMeans(out$lambda))

hist(out$phi)
(apply(out$lambda, MARGIN = 2, sd) / apply(out_pos$lambda, MARGIN = 2, FUN = sd)) %>% hist()

## Testing the new framework ----

probs <- Matrix::Matrix(diag(ncol(X))) ## Sparsity
qpois_forest <- MakeQPois(probs = probs, ## Make tree
                          num_tree = num_tree,
                          k = 1.5,
                          update_s = FALSE,
                          phi = 1)

pb <- progress::progress_bar$new(
  format = "  running :what [:bar] :percent eta: :eta",
  clear = FALSE, total = 1000, width = 60)
stuff <- matrix(nrow = 1000, ncol = nrow(X))

for(i in 1:1000) {
  pb$tick()
  stuff[i,] <- as.numeric(qpois_forest$do_gibbs(X, Y, 
                                                rep(0, nrow(X)), X, 1)[[1]])
}


