library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)
library(tidyverse)

P <- 10
N <- 250
sigma <- 20
num_tree <- 50

sim_fried_pois <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  lambda <- sigma * f(X)
  Y <- rpois(n = n, lambda = lambda)
  
  return(list(X = X, Y = Y, lambda = lambda))
}

c(X,Y,lambda) %<-% sim_fried_pois(N,P,sigma)
probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)

system.time({
  out <- PoisBart(X, Y, probs, num_tree, scale_lambda = 1, 
                  scale_lambda_0 = 1 / sqrt(num_tree), 
                  num_burn = 1000, num_thin = 1, num_save = 1000)  
})


plot(log(lambda), colMeans(out$lambda))
rmse(log(lambda), colMeans(out$lambda))
abline(a=0,b=1)


