## Load ----

library(Batman)
library(tidyverse)
library(zeallot)

## Generate data ----

N     <- 300
P     <- 10
sigma <- 2
rho   <- 3
phi_0 <- 1 / (rho + 1)

sim_beta <- function(N, P, sigma) {
  X <- matrix(runif(N * P), nrow = N)
  logit_mu <- 10 * sin(pi * X[,1] * X[,2]) + 20 * (X[,3] - 0.5)^2 +
    10 * X[,4] + 5 * X[,5]
  logit_mu <- (logit_mu - 14) / 5 * sigma
  mu <- plogis(logit_mu)
  alpha <- mu * rho
  beta <- (1 - mu) * rho
  Y <- rbeta(N, alpha, beta)
  return(list(X = X, Y = Y, mu = mu, logit_mu = logit_mu))
}

c(X, Y, mu, logit_mu) %<-% sim_beta(N, P, sigma)

plot(mu, Y)
plot(density(Y))
plot(\(x) dbeta(x, rho * mu[1], rho * (1 - mu[1])))

## Fit Model ----

fitted_qbeta <- QBinomBart(
  X = X,
  Y = Y,
  n = rep(1, N),
  X_test = X,
  probs = Matrix::Matrix(diag(P)),
  num_trees = 50,
  scale_lambda = 1 / sqrt(50),
  scale_lambda_0 = 1,
  num_burn = 1000,
  num_thin = 1,
  num_save = 1000
)
