## Load ------------------------------------------------------------------------

library(Batman)
library(tidyverse)

## Generate Data ---------------------------------------------------------------

set.seed(123)

# Set parameters
n_samples <- 999
n_features <- 5
n_categories <- 3


# Generate random features
X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)
X <- rbind(0, X)
QX <- SoftBart::quantile_normalize_bart(X)
QXt <- QX[1,,drop=FALSE]
QX <- QX[-1,]
Xt <- X[1,]
X <- X[-1,]

# Generate random coefficients, including intercept
# beta <- 0 * rnorm(n_features)
beta <- rep(1 / sqrt(n_features), n_features)
gamma <- c(-2,1)

linear_predictor <- X %*% beta

# Transform linear predictor using the complementary log-log link function
# and then to probabilities

## Truth --- PH
p_0 <- 1 - exp(-exp(gamma[1] + linear_predictor))
p_1 <- exp(-exp(gamma[1] + linear_predictor)) *
  (1 - exp(-exp(gamma[2] + linear_predictor)))
p_2 <- exp(-exp(gamma[1] + linear_predictor)) *
  exp(-exp(gamma[2] + linear_predictor))

par(mfrow=c(2,2))
p <- cbind(p_0, p_1, p_2)
hist(p_0)
hist(p_1)
hist(p_2)

## Truth --- NPH
X_2 <- cbind(X, rep(0.5, nrow(X)))
beta_2 <- c(beta, 0.25)

p_0 <- 1 - exp(-exp(gamma[1] + linear_predictor))
p_1 <- exp(-exp(gamma[1] + linear_predictor)) * 
  (1 - exp(-exp(gamma[2] + X_2 %*% beta_2)))
p_2 <- exp(-exp(gamma[1] + linear_predictor)) * 
  exp(-exp(gamma[2] + X_2 %*% beta_2))

par(mfrow=c(2,2))
p <- cbind(p_0, p_1, p_2)
hist(p_0)
hist(p_1)
hist(p_2)

## Get Outcomes
Y <- sapply(1:nrow(QX), \(i) sample(1:3, 1, prob = p[i,]))

## Fit model -------------------------------------------------------------------

#args(CoxNPHOBart)

QX2 <- cbind(QX, 0)
temp <- c(diag(n_features + 1)) 
temp[length(temp)] <- 0.25
s <- Matrix::Matrix(temp, n_features + 1, n_features + 1)
#temp <- c(rep(1, n_features), 0, rep(0, n_features), 1)
#s <- Matrix::Matrix(temp, n_features + 1, 2, sparse = F)
#s   <- Matrix::Matrix(diag(ncol(QX2)))
bin_to_list <- lapply(1:max(Y), function(i) which(Y == i) - 1)

my_fit <-
  CoxNPHOBart(
    QX2,
    Y - 1,
    bin_to_list,
    probs = s,
    X_test = QX2,
    num_trees = 50,
    scale_lambda = 2 / sqrt(50),
    shape_gamma = 1,
    rate_gamma = 1,
    num_burn = 1500,
    num_thin = 1,
    num_save = 1500
  )

# p_0_samps <- t(1 - exp(-exp(t(my_fit$lambda_test[,1,]) + my_fit$gamma[,1])))
# p_1_samps <- t(1 - exp(-exp(t(my_fit$lambda_test[,2,]) + my_fit$gamma[,2]))) * 
#   t(exp(-exp(t(my_fit$lambda_test[,1,]) + my_fit$gamma[,1])))
# p_0_hat <- rowMeans(p_0_samps)
# p_1_hat <- rowMeans(p_1_samps)

p_0_samps <- 1 - exp(- exp(my_fit$lambda_train[,1,] + my_fit$gamma[,1]))
p_1_samps <- (1 - exp(-exp(my_fit$lambda_train[,2,] + my_fit$gamma[,2]))) * 
  exp(-exp((my_fit$lambda_train[,1,] + my_fit$gamma[,1])))
p_0_hat <- rowMeans(p_0_samps)
p_1_hat <- rowMeans(p_1_samps)


plot(p_0_hat %>% log, p_0 %>% log)
abline(a=0,b=1,col='green')
plot(p_1_hat %>% log, p_1 %>% log)
abline(a=0,b=1,col='green')

plot(p_hat_0 %>% log, p_0 %>% log, cex = .2, col = 'blue')
abline(a=0,b=1,col='green')
plot((p_hat_1 %>% log), p_1 %>% log)
abline(a=0,b=1,col='green')

mean((log(p_hat_0) - log(p_0))^2)
mean((log(p_0_hat) - log(p_0))^2)
mean((log(p_hat_1) - log(p_1))^2)
mean((log(p_1_hat) - log(p_1))^2)

# ## Call Function ---------------------------------------------------------------
# 
# # set.seed(883293)
# 
# out <- CLogLogOrdinalBart(
#   X = QX,
#   Y = Y - 1,
#   num_levels = 3,
#   X_test = QXt,
#   probs = Matrix::Matrix(diag(ncol(X))),
#   num_trees = 50,
#   scale_lambda = 2 / sqrt(50),
#   alpha_gamma = 2,
#   beta_gamma = 2,
#   gamma_0 = log(-log(mean(Y > 1))),
#   # gamma_0 = mean(moo[,1]),
#   num_burn = 1000,
#   num_thin = 1,
#   num_save = 1000
# )
# 
# p_hat_0 <- colMeans(1 - exp(-exp(out$lambda + out$gamma[,1])))
# p_hat_1 <- colMeans((1 - exp(-exp(out$lambda + out$gamma[,2]))) * exp(-exp(out$lambda + out$gamma[,1])))
# p_hat_2 <- 1 - p_hat_1 - p_hat_0
# 
# par(mfrow = c(3,2), mar = c(5,4,1,1))
# colMeans(out$gamma)
# moo <- out$gamma + rowMeans(out$lambda)
# plot(moo[,1])
# abline(h = gamma[1] + mean(linear_predictor))
# plot(moo[,2])
# abline(h = gamma[2] + mean(linear_predictor))
# plot(out$gamma[,2])
# plot(colMeans(out$lambda) - mean(out$lambda), linear_predictor)
# abline(a=0,b=1)
# plot(out$sigma_mu)
# plot(p_hat_2, p_2)
# abline(a=0,b=1,col='red',lwd=3)
# 
# ## Individual stuff ------------------------------------------------------------
# 
# ps_0 <- 1 - exp(-exp(out$lambda_test + out$gamma[,1]))
# ps_1 <- exp(-exp(out$lambda_test + out$gamma[,1])) * (1 - exp(-exp(out$lambda_test + out$gamma[,2])))
# 
# plot(p_hat_0, p_0)
# abline(a=0,b=1,col='red',lwd=3)
# plot(p_hat_1, p_1)
# abline(a=0,b=1,col='red',lwd=3)
# 
# plot(log(-log(1-ps_0)))
# abline(h = gamma[1], col = 3, lwd = 3)
# 
# plot(rowMeans(out$lambda + out$gamma[,1]))
# abline(h = gamma[1] + mean(linear_predictor), col = 3)

