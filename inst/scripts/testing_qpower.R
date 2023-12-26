## Load ----

library(Batman)
library(dbarts)
library(tidyverse)

rmse <- function(x,y) sqrt(mean(abs(x-y)^2))

## Generate data ----

# set.seed(999)

P <- 10
N <- 500
num_tree <- 50

probs <- Matrix::Matrix(diag(P))

sim_fried_gamma <- function(N, P, phi = 0.5) {
  X <- matrix(runif(N*P), nrow = N)
  
  f <- function(x)
    exp(10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 5 * x[,5])^(1/10)
  
  phi <- 0.5
  mu <- f(X)
  alpha <- 1 / phi
  beta <- alpha / mu
  
  Y <- rgamma(n = N, shape = alpha, rate = beta)
  
  return(data.frame(X = X, Y = Y, mu = mu))
}

train_data <- sim_fried_gamma(N, P)
qplot(mu, Y, data = train_data)

X <- train_data %>% dplyr::select(-Y, -mu) %>% as.matrix()
Y <- train_data %>% pull(Y)

## Fit Model ----

fitted_qpower <-
  QPowerBart(
    X,
    Y,
    X,
    probs,
    50,
    scale_lambda_0 = 1,
    scale_lambda = 1 / sqrt(50),
    num_burn = 1000,
    num_thin = 1,
    num_save = 1000
  )

par(mfrow = c(1,3))
hist(fitted_qpower$p)
hist(fitted_qpower$phi)
plot(rowSums(fitted_qpower$counts))

## Comparing with qgamma ----

fitted_qgamma <-
  QGammaBart(
    X,
    Y,
    X,
    probs,
    50,
    scale_lambda_0 = 1,
    scale_lambda = 1 / sqrt(50),
    num_burn = 1000,
    num_thin = 1,
    num_save = 1000
  )

## Plotting results ----

par(mfrow = c(1,2))

mu_hat_qgam <- -fitted_qgamma$lambda %>% exp() %>% colMeans()
mu_hat_qpow <- fitted_qpower$lambda %>% exp() %>% colMeans()
mu_0 <- train_data$mu

rmse(mu_0, mu_hat_qgam)
rmse(mu_0, mu_hat_qpow)

plot(mu_hat_qgam, mu_0)
plot(mu_hat_qpow, mu_0)

# fitted_dbart <-
#   bart(
#     x.train = X,
#     y.train = Y,
#     x.test = X,
#     ntree = 50,
#     nskip = 3000,
#     ndpost = 3000
#   )

