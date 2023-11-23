## Load ----

library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)
library(tidyverse)
library(matrixStats)

P <- 10
N <- 250
num_tree <- 50
num_burn <- 4000
num_thin <- 1
num_save <- 4000

## Sim ----

sigma <- 0.1
set.seed(1234)
sim_fried_mlogit <- function(n, p, sigma, phi = 0.5) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  s <- 1 / phi - 1
  X <- matrix(runif(n*p), nrow = n)
  p <- pnorm(sigma * (f(X) - 12))
  Y <- rbeta(n = n, shape1 = s * p, shape2 = s * (1 - p))
  n <- rep(1, length(Y))
  
  return(list(X = X, Y = Y, p = p, n = n))
}

c(X,Y,p,n) %<-% sim_fried_mlogit(N,P,sigma)
probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)

## Fit ----

quasi_fit <-
  QBinomBart(
    X = X,
    Y = Y,
    n = n,
    X_test = X,
    probs = probs,
    num_trees = num_tree,
    scale_lambda = 1.5 / sqrt(num_tree),
    scale_lambda_0 = 1,
    num_burn = num_burn,
    num_thin = num_thin,
    num_save = num_save
  )

## Evaluate Output ----

p_1 <- plogis(quasi_fit$lambda)

par(mfrow = c(1,2))
plot(colMeans(p_1), p)
hist(quasi_fit$phi)
par(mfrow = c(1,1))
