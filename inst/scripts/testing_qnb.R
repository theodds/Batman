## Load ----

library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)
library(tidyverse)

rmse <- function(x,y) sqrt(mean((x-y)^2))

## Generate quasi-Poisson Data just to test; expect k in this case to be close to 0 (hopefully not negative) ----

P <- 10
N <- 2000
sigma <- 20
num_tree <- 50

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

## Fit Quasi-NB ----

qnb_fit <- QNBBart(
  X = X,
  Y = Y,
  X_test = X,
  probs = probs,
  num_trees = num_tree,
  scale_lambda = 1 / sqrt(num_tree),
  scale_lambda_0 = 1,
  num_burn = 0,
  num_thin = 1,
  num_save = 1000
)

plot(qnb_fit$phi)

## Fit Quasi-Poisson ----

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