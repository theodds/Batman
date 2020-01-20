# Load --------------------------------------------------------------------


library(Batman)
library(dbarts)
library(zeallot)
library(Matrix)

P <- 10
N <- 250
num_tree <- 50


# Sim ---------------------------------------------------------------------

sigma <- 2
set.seed(1234)
sim_fried_mlogit <- function(n,p,sigma) {
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  X <- matrix(runif(n*p), nrow = n)
  p <- pnorm(sigma * (f(X) - 12))
  Y <- rbinom(n = n, size = 1, prob = p)
  
  return(list(X = X, Y = Y, p = p))
}


c(X,Y,p) %<-% sim_fried_mlogit(N,P,sigma)
mean(Y)
probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)


# Fit ---------------------------------------------------------------------
time_out <- system.time({
  out <- MLogitBart(X = X, Y = Y, probs = probs, num_cat = 2, num_trees = num_tree, 
                    scale_lambda = .2, 
                    shape_lambda_0 = .1, 
                    rate_lambda_0 = .1, 
                    num_burn = 3000, 
                    num_thin = 1, 
                    num_save = 3000)  
})


dbarts_bin <- bart(x.train = X, y.train = Y, ntree = num_tree, 
                   ndpost = 4000, nskip = 4000)

plot(p, rowMeans(out$pi[,2,]))
plot(p, colMeans(pnorm(dbarts_bin$yhat.train)))

p_hat_1 <- rowMeans(out$pi[,2,])
p_hat_2 <- colMeans(pnorm(dbarts_bin$yhat.train))

-2*sum(p * log(p_hat_1) + (1 - p) * log(1 - p_hat_1))
-2*sum(p * log(p_hat_2) + (1 - p) * log(1 - p_hat_2))
print(time_out)
