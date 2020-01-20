# Load --------------------------------------------------------------------

library(Batman)
library(zeallot)
library(rbart)
library(Matrix)

# Sim ---------------------------------------------------------------------

set.seed(7658)

P <- 10
N <- 250
num_tree <- 50

sim_data <- function(n,p, sigma = 2) {
  X <- matrix(runif(n * p), nrow = n)
  mu <- 4 * X[,1]^2
  sigma <- 0.2 * exp(2 * X[,1])
  Y_var <- mu + sigma * rnorm(n)
  
  f <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 
    20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  p <- pnorm(sigma * (f(X) - 12))
  Y_logit <- rbinom(n = n, size = 1, prob = p)
  
  return(list(X = X, Y_var = Y_var, Y_logit = Y_logit,  
              mu = mu, sigma = sigma, p = p))
}

c(X, Y_var, Y_logit, mu, sigma, p) %<-% sim_data(N, P)

loc_y <- mean(Y_var)
scale_y <- sd(Y_var)
Y_scale <- (Y_var - loc_y) / scale_y

probs <- diag(P); probs <- Matrix(probs, sparse = TRUE)

# Fit ---------------------------------------------------------------------

out <- VarLogitBart(X_logit             = X, 
                    Y_logit             = Y_logit, 
                    X_var               = X, 
                    Y_var               = Y_scale, 
                    probs               = probs, 
                    num_cat             = 2, 
                    num_trees           = num_tree, 
                    scale_lambda        = 1/sqrt(num_tree), 
                    shape_lambda_0      = 0.1, 
                    rate_lambda_0       = 0.1, 
                    scale_kappa         = 1/sqrt(num_tree), 
                    sigma_scale_log_tau = 0.1/sqrt(num_tree), 
                    shape_tau_0         = 0.1, 
                    rate_tau_0          = 0.1,
                    num_burn            = 1000, 
                    num_thin            = 1, 
                    num_save            = 1000)


rmse <- function(x,y) sqrt(mean((x-y)^2))
mu_hat <- colMeans(out$mu) * scale_y + loc_y
rmse(mu_hat, mu)

plot(colMeans(1/sqrt(out$tau)) * scale_y, sigma)
abline(a=0,b=1)

plot(rowMeans(out$pi[,2,]), p)

