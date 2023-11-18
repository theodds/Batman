## Load ----

library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)

P <- 10
N <- 250
num_tree <- 50
num_within <- 10
num_burn <- 100
num_thin <- 1
num_save <- 100

## Sim ----

sigma <- 2
set.seed(1234)
sim_fried_mlogit <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  p <- pnorm(sigma * (f(X) - 12))
  Y <- rbinom(n = n, size = num_within, prob = p)
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

## Fitting dbarts ----

dbarts_bin <- bart(x.train = X, y.train = Y, ntree = num_tree, 
                   ndpost = 4000, nskip = 4000)

plot(p, rowMeans(out$pi[,2,]))
plot(p, colMeans(pnorm(dbarts_bin$yhat.train)))

p_hat_1 <- rowMeans(out$pi[,2,])
p_hat_2 <- colMeans(pnorm(dbarts_bin$yhat.train))

-2*sum(p * log(p_hat_1) + (1 - p) * log(1 - p_hat_1))
-2*sum(p * log(p_hat_2) + (1 - p) * log(1 - p_hat_2))
print(time_out)
