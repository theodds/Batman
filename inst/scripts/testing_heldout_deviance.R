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

sim_fried_ig <- function(N, P, phi = 0.5) {
  X <- matrix(runif(N*P), nrow = N)
  
  f <- function(x)
    exp(10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 10 * x[,4] + 
        5 * x[,5])^(1/10)
  
  mu      <- f(X)
  alpha   <- 2 + 1 / phi
  epsilon <- 1 / rgamma(N, alpha, alpha - 1)
  Y       <- f(X) * epsilon
  
  return(data.frame(X = X, Y = Y, alpha = alpha, mu = mu))
}

train_data <- sim_fried_ig(N, P)
test_data <- sim_fried_ig(N, P)

X <- train_data %>% select(-Y, -mu) %>% as.matrix()
Y <- train_data %>% pull(Y)
X_test <- test_data %>% select(-Y, -mu) %>% as.matrix()

## Fit Model ----

fitted_qgam <-
  QGammaBart(
    X,
    Y,
    X_test,
    probs,
    50,
    scale_lambda_0 = 1,
    scale_lambda = 1 / sqrt(50), 
    phi_update = 2,
    num_burn = 3000,
    num_thin = 1,
    num_save = 3000
  )


fitted_qgam_oracle <- GammaRegBart(X = X[,1:5], Y = Y, X_test = X_test[,1:5], probs = probs[1:5,1:5],
                                   num_trees = 50, scale_lambda_0 = 1,
                                   scale_lambda = 1 / sqrt(50), num_burn = 3000,
                                   num_thin = 1, num_save = 3000)

fitted_gam <- GammaRegBart(X = X, Y = Y, X_test = X_test, probs = probs,
                           num_trees = 50, scale_lambda_0 = 1,
                           scale_lambda = 1 / sqrt(50), num_burn = 3000,
                           num_thin = 1, num_save = 3000)

fitted_gam_oracle <- GammaRegBart(X = X[,1:5], Y = Y, X_test = X_test[,1:5], probs = probs[1:5, 1:5],
                                  num_trees = 50, scale_lambda_0 = 1,
                                  scale_lambda = 1 / sqrt(50), num_burn = 3000,
                                  num_thin = 1, num_save = 3000)

## Function for computing the deviance per iteration ----

calc_deviance <- function(y, mu) {
  -2 - 2 * log(y) + 2 * y / mu + 2 * log(mu)
}

calc_loglik <- function(y, mu, alpha) {
  dgamma(y, shape = alpha, rate = alpha / mu, log = TRUE)
}

calc_loo_devs_gamma <- function(fit) {
  mu <- exp(-fit$lambda)
  ham <- sapply(1:nrow(mu), \(i) calc_loglik(Y, mu[i,], fit$alpha[i])) %>% t()
  d2 <- apply(mu, 1, \(x) calc_deviance(Y, x)) %>% t()
  w <- exp(-ham)
  loo_dev <- colMeans(d2 * w) / colMeans(w)
}

calc_loo_devs <- function(fit) {
  mu <- exp(-fit$lambda)
  ham <- apply(mu, 1, \(x) calc_deviance(Y, x)) %>% t()
  w <- exp(ham / as.numeric(fitted_qgam$phi) / 2)
  loo_dev <- colMeans(ham * w) / colMeans(w)

  return(loo_dev)
}

loo_dev_gam        <- calc_loo_devs_gamma(fitted_gam)
loo_dev_qgam       <- calc_loo_devs(fitted_qgam)
loo_dev_oracle     <- calc_loo_devs(fitted_qgam_oracle)
loo_dev_gam_oracle <- calc_loo_devs_gamma(fitted_gam_oracle)

data.frame(Method = c("Gam", "QGam", "Gam Oracle", "QGam Oracle"),
           LOO = c(sum(loo_dev_gam), sum(loo_dev_qgam),
                   sum(loo_dev_gam_oracle), sum(loo_dev_oracle)))
wilcox.test(loo_dev_qgam - loo_dev_gam, alternative = "greater")
## hist(fitted_qgam$phi)

## ## Comparing with dbarts ----

## fitted_dbart <-
##   bart(
##     x.train = X,
##     y.train = Y,
##     x.test = X,
##     ntree = 50,
##     nskip = 3000,
##     ndpost = 3000
##   )

## ## Checking output ----

## par(mfrow = c(3,3))
## plot(colMeans(exp(-fitted_qgam$lambda)), train_data$mu)
## abline(a=0, b=1, lwd = 3, lty = 2, col = 2)
## hist(fitted_qgam$phi)
## plot(fitted_dbart$yhat.test.mean, train_data$mu)
## abline(a=0, b=1, lwd = 3, lty = 2, col = 2)
## hist(fitted_dbart$sigma)
## plot(fitted_qgam$phi, type = 'l')
## plot(colMeans(fitted_qgam$counts > 0), ylim = c(0,1))

## abs_errors <- abs(colMeans(exp(-fitted_qgam$lambda)) - train_data$mu)^2
## abs_errors_db <- abs(fitted_dbart$yhat.test.mean - train_data$mu)^2
## plot(density(abs_errors))
## lines(density(abs_errors_db))
## plot(fitted_qgam$counts[,1])
## plot(fitted_qgam$counts[,2])

## mean(abs_errors) / mean(abs_errors_db)

