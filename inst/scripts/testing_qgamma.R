## Load ----

library(Batman)
library(tidyverse)
library(dbarts)

## Generate data ----

set.seed(999)

P <- 10
N <- 2500
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
    num_burn = 3000,
    num_thin = 1,
    num_save = 3000
  )

## Comparing with dbarts ----

fitted_dbart <-
  bart(
    x.train = X,
    y.train = Y,
    x.test = X,
    ntree = 50,
    nskip = 3000,
    ndpost = 3000
  )

## Checking output ----

par(mfrow = c(3,3))
plot(colMeans(exp(-fitted_qgam$lambda)), train_data$mu)
abline(a=0, b=1, lwd = 3, lty = 2, col = 2)
hist(fitted_qgam$phi)
plot(fitted_dbart$yhat.test.mean, train_data$mu)
abline(a=0, b=1, lwd = 3, lty = 2, col = 2)
hist(fitted_dbart$sigma)
plot(fitted_qgam$phi, type = 'l')
plot(colMeans(fitted_qgam$counts > 0), ylim = c(0,1))

abs_errors <- abs(colMeans(exp(-fitted_qgam$lambda)) - train_data$mu)^2
abs_errors_db <- abs(fitted_dbart$yhat.test.mean - train_data$mu)^2
plot(density(abs_errors))
lines(density(abs_errors_db))

mean(abs_errors) / mean(abs_errors_db)

