## Load ----

library(Batman)
library(tidyverse)

## Generate data ----

set.seed(9999)

P <- 10
N <- 250
num_tree <- 50

probs <- Matrix::Matrix(diag(P))

sim_fried_gamma <- function(N, P, phi = 0.5) {
  X <- matrix(runif(N*P), nrow = N)
  
  f <- function(x)
    10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5]
  
  phi <- 0.5
  mu <- f(X)
  alpha <- 1 / phi
  beta <- alpha / mu
  
  Y <- rgamma(n = N, shape = alpha, rate = beta)
  
  return(data.frame(X = X, Y = Y, mu = mu))
}

train_data <- sim_fried_gamma(N, P)
qplot(mu, Y, data = train_data)

X <- train_data %>% select(-Y, -mu) %>% as.matrix()
Y <- train_data %>% pull(Y)

## Fit Model ----

fitted_qgam <-
  QGammaBart(
    X,
    Y,
    X,
    probs,
    50,
    scale_lambda_0 = 1,
    scale_lambda = 1 / sqrt(50),
    num_burn = 10,
    num_thin = 1,
    num_save = 10
  )
