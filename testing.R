library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)

P <- 10
N <- 250
sigma <- 1
num_tree <- 50

sim_fried <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]

  X <- matrix(runif(n*p), nrow = n)
  mu <- f(X)
  Y <- mu + sigma * rnorm(n)

  return(list(X = X, Y = Y, mu = mu))
}

c(X,Y,mu) %<-% sim_fried(N,P,sigma)
probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)
mu_Y <- mean(Y)
sd_Y <- sd(Y) 
Y_norm <- (Y - mu_Y) / sd_Y


out <- RegBart(X, Y_norm, probs, num_tree, 1/sd_Y, 1.5/sqrt(num_tree), 1000, 1, 1000)
dbart_out <- bart(x.train = X, y.train = Y)
# 
rmse <- function(x,y) sqrt(mean((x-y)^2))
rmse(colMeans(out$mu) * sd_Y + mu_Y, mu)
rmse(dbart_out$yhat.train.mean, mu)
colMeans(out$counts)
