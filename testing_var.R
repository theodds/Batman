# Load --------------------------------------------------------------------

library(Batman)
library(zeallot)
library(rbart)
library(Matrix)

# Sim ---------------------------------------------------------------------

set.seed(7658)

num_tree <- 50

sim_data <- function(n,p) {
  X <- matrix(runif(n * p), nrow = n)
  mu <- 4 * X[,1]^2
  sigma <- 0.2 * exp(2 * X[,1])
  Y <- mu + sigma * rnorm(n)
  return(list(X = X, Y = Y, mu = mu, sigma = sigma))
}

c(X,Y,mu,sigma) %<-% sim_data(500, 1)
mu_Y <- mean(Y)
sigma_Y <- sd(Y)
Y_scale <- (Y - mu_Y) / sigma_Y
## probs <- Matrix(data = c(1,0,0,1), nrow = 2, byrow = TRUE, sparse = TRUE)
probs <- Matrix(data = 1, nrow = 1, byrow = TRUE, sparse = TRUE)

empirical_lm <- lm(Y_scale ~ X)
empirical_sigma <- sqrt(mean(empirical_lm$residuals^2))

out <- VarBart(X, Y_scale, probs, 1.5 / sqrt(num_tree), .1 / sqrt(num_tree), 3, 3, num_tree, 4000, 1, 4000)
outr <- rbart(X, Y, ntree = 50, ntreeh = 50, ndpost = 4000, nskip = 4000)

pred_outr <- predict(outr)
pred_outr$mmean
pred_outr$smean

mu_hat <- colMeans(out$mu) * sigma_Y + mu_Y
sigma_hat <- colMeans(1/sqrt(out$tau)) * sigma_Y



par(mfrow = c(1,2))

plot(X, mu)
o <- order(X)
lines(X[o], mu_hat[o])
lines(X[o], pred_outr$mmean[o], col = 'red')

plot(X, sigma)
lines(X[o], sigma_hat[o])
lines(X[o], pred_outr$smean[o], col = 'red')

mean(abs(log(sigma_hat) - log(sigma)))
mean(abs(log(pred_outr$smean) - log(sigma)))

mean(abs(mu_hat - mu))
mean(abs(pred_outr$mmean - mu))

