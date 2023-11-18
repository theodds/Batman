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
num_within <- 10
num_burn <- 4000
num_thin <- 1
num_save <- 4000

## Sim ----

sigma <- 0.1
set.seed(1234)
sim_fried_mlogit <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  p <- pnorm(sigma * (f(X) - 12))
  Y <- 2 * rbinom(n = n, size = num_within, prob = p / 2)
  n <- rep(num_within, length(Y))
  
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

## Prepping data for dbarts ----

convert_y <- function(x, y, n) {
  yy <- numeric(n)
  yy[1:y] <- 1
  xx <- matrix(nrow = 0, ncol = length(x))
  for(i in 1:length(x)) {
    xx <- rbind(xx, x)
  }
  return(data.frame(X = xx, Y = yy))
}

expand_df <- map_df(.x = 1:nrow(X), .f = \(i) convert_y(X[i,], Y[i], n[i]))

X_expand <- as.matrix(expand_df %>% select(-Y))
Y_expand <- expand_df$Y

## Fitting dbarts ----

dbarts_bin <- bart(x.train = X_expand, y.train = Y_expand, ntree = num_tree, 
                   ndpost = 4000, nskip = 4000)

p_hat_1 <- colMeans(plogis(quasi_fit$lambda))
p_hat_2 <- colMeans(pnorm(dbarts_bin$yhat.train[, seq(1,nrow(X_expand),n[1])]))

plot(p, p_hat_1)
plot(p, p_hat_2)

-2*sum(p * log(p_hat_1) + (1 - p) * log(1 - p_hat_1))
-2*sum(p * log(p_hat_2) + (1 - p) * log(1 - p_hat_2))

## Comparing standard errors ----

p_1 <- plogis(quasi_fit$lambda)
p_2 <- pnorm(dbarts_bin$yhat.train[, seq(1, nrow(X_expand), n[1])])

par(mfrow = c(1,2))
plot(colSds(p_1), colSds(p_2))
plot(density(colSds(p_1)), lwd = 2)
lines(density(colSds(p_2)), col = 3, lty = 2, lwd = 2)
par(mfrow = c(1,1))
