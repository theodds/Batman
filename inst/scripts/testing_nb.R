## Load ------------------------------------------------------------------------

library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)
library(tidyverse)
library(patchwork)

rmse <- function(x,y) sqrt(mean((x-y)^2))

## Generate NB Data just to test with k = 2 ------------------------------------

P <- 10
N <- 2000
sigma <- 2
num_tree <- 50

sim_fried_nb <- function(n,p,sigma,k=2) {
  f <- function(x)
    (10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5] - 14) / 5
  
  X <- matrix(runif(n*p), nrow = n)
  lambda <- exp(sigma * f(X))
  # alpha <- rgamma(n = n, shape = k, rate = k)
  xi <- rgamma(n = n, shape = k, rate = k / lambda)
  Y <- rpois(n = n, lambda = xi)
  
  return(list(X = X, Y = Y, lambda = lambda, k = k))
}

# set.seed(20939)
my_data <- sim_fried_nb(N, P, sigma)

MASS::glm.nb(my_data$Y ~ I(log(my_data$lambda))) %>% summary()

## Fit Quasi-NB ----------------------------------------------------------------

X <- my_data$X
Y <- my_data$Y
probs <- Matrix::Matrix(diag(ncol(X)))
num_tree <- 50

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

p_1 <- qplot(1 / sqrt(qnb_fit$k)) + geom_vline(xintercept = 1 / sqrt(my_data$k))
p_2 <- qplot(colMeans(qnb_fit$lambda), log(my_data$lambda)) + geom_abline(slope = 1, intercept = 0, color = 'steelblue')

p_1 + p_2




## Junk ------------------------------------------------------------------------

# 
# phi <- 2
# k <- 2
# sim_fried_qnb <- function(n,p,sigma) {
#   f <- function(x)
#     10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
#   
#   X <- matrix(runif(n*p), nrow = n)
#   lambda <- sigma * f(X)
#   Y1 <- phi * rpois(n = n, lambda = lambda / phi)
#   Y2 <- rnbinom(n = n, size = k, mu = lambda)
#   Y <- ifelse(runif(n) < 0.5, Y1, Y2)
#   
#   return(list(X = X, Y = Y, lambda = lambda))
# }
# 
# phi_0 <- (phi + 1) / 2
# k_0 <- (phi + 1) * k
# 
# # c(X,Y,lambda) %<-% sim_fried_pois(N,P,sigma)
# c(X,Y,lambda) %<-% sim_fried_qnb(N,P,sigma)
# mean((Y - lambda)^2 / phi_0 / (lambda + lambda^2 / k_0))
# probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)
# 

# 
# par(mfrow = c(1,3))
# plot(qnb_fit$phi)
# plot(qnb_fit$k)
# plot(colMeans(qnb_fit$lambda), log(lambda))
# abline(a=0,b=1)
# 
# ## Testing By Hand ----
# 
# pl <- function(theta) {
#   phi <- theta[1]
#   k <- theta[2]
#   sigma_sq <- phi * (lambda + lambda^2 / k)
#   return(-sum(dnorm(Y, lambda, sqrt(sigma_sq), log = TRUE)))
# }
# 
# test_df <- optim(c(1,1), pl, method = "SANN")
# print(test_df)
# 
# ## Fit Quasi-Poisson ----
# 
# system.time({
#   out <- QPoisBart(X, Y, X_test = X, probs = probs, num_trees = num_tree, scale_lambda = 1, 
#                    scale_lambda_0 = 1 / sqrt(num_tree),
#                    num_burn = 2000, num_thin = 1, num_save = 2000)  
# })
# 
# plot(colMeans(out$lambda), log(lambda))
# abline(a=0,b=1)
# 
# plot(log(lambda), colMeans(out_pos$lambda))
# rmse(log(lambda), colMeans(out_pos$lambda))
# rmse(log(lambda), colMeans(out$lambda))
# 
# hist(out$phi)
# (apply(out$lambda, MARGIN = 2, sd) / apply(out_pos$lambda, MARGIN = 2, FUN = sd)) %>% hist()