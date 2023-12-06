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

phi <- 2
k <- 2
sim_fried_qnb <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  lambda <- sigma * f(X)
  Y1 <- phi * rpois(n = n, lambda = lambda / phi)
  Y2 <- rnbinom(n = n, size = k, mu = lambda)
  Y <- ifelse(runif(n) < 0.5, Y1, Y2)
  
  return(list(X = X, Y = Y, lambda = lambda))
}

phi_0 <- (phi + 1) / 2
k_0 <- (phi + 1) * k

# c(X,Y,lambda) %<-% sim_fried_pois(N,P,sigma)
c(X,Y,lambda) %<-% sim_fried_qnb(N,P,sigma)
mean((Y - lambda)^2 / phi_0 / (lambda + lambda^2 / k_0))
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
  num_burn = 1000,
  num_thin = 1,
  num_save = 1000
)

par(mfrow = c(1,3))
plot(qnb_fit$phi)
plot(qnb_fit$k)
plot(colMeans(qnb_fit$lambda), log(lambda))
abline(a=0,b=1)

## Testing By Hand ----

pl <- function(theta) {
  phi <- theta[1]
  k <- theta[2]
  sigma_sq <- phi * (lambda + lambda^2 / k)
  return(-sum(dnorm(Y, lambda, sqrt(sigma_sq), log = TRUE)))
}

test_df <- optim(c(1,1), pl, method = "SANN")
print(test_df)

## Fit Quasi-Poisson ----

system.time({
  out <- QPoisBart(X, Y, X_test = X, probs = probs, num_trees = num_tree, scale_lambda = 1, 
                   scale_lambda_0 = 1 / sqrt(num_tree),
                   num_burn = 2000, num_thin = 1, num_save = 2000)  
})

plot(colMeans(out$lambda), log(lambda))
abline(a=0,b=1)

plot(log(lambda), colMeans(out_pos$lambda))
rmse(log(lambda), colMeans(out_pos$lambda))
rmse(log(lambda), colMeans(out$lambda))

hist(out$phi)
(apply(out$lambda, MARGIN = 2, sd) / apply(out_pos$lambda, MARGIN = 2, FUN = sd)) %>% hist()