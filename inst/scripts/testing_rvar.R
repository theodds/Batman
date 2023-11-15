## Load ----

library(Batman)
library(zeallot)
library(rbart)
library(Matrix)

## Sim ----

set.seed(7658)

num_tree <- 50

sim_data <- function(n,p) {
  X <- matrix(runif(n * p), nrow = n)
  mu <- rep(0,n)
  sigma <- 0.2 * exp(2 * X[,1])
  Y <- mu + sigma * rnorm(n)
  return(list(X = X, Y = Y, mu = mu, sigma = sigma))
}

c(X,Y,mu,sigma) %<-% sim_data(500, 1)

## Prepare data for fitting ----

sigma_Y         <- sd(Y)
Y_scale         <- Y / sigma_Y
probs           <- Matrix(data = 1, nrow = 1, byrow = TRUE, sparse = TRUE)
empirical_lm    <- lm(Y_scale ~ X)
empirical_sigma <- sqrt(mean(empirical_lm$residuals^2))

## Fit the model, and rbart for comparison ----

out <- RVarBart(X = X, Y = Y_scale,
                probs = probs,
                sigma_scale_log_tau = 1.5 / sqrt(num_tree),
                shape_tau_0 = 3,
                rate_tau_0 = 3,
                num_trees = num_tree,
                num_burn = 4000,
                num_thin = 1,
                num_save = 4000)

outr <- rbart(X, Y, ntree = 50, ntreeh = 50, ndpost = 4000, nskip = 4000)

## Collect predictions for rbart and RVarBart ----

pred_outr <- predict(outr)
sigma_hat <- colMeans(1/sqrt(out$tau)) * sigma_Y

## Plot for comparison ----

par(mfrow = c(1,2))

plot(X, sigma)
lines(X[o], sigma_hat[o])
lines(X[o], pred_outr$smean[o], col = 'red')

## Measure Accuracy ----

mean(abs(log(sigma_hat) - log(sigma)))
mean(abs(log(pred_outr$smean) - log(sigma)))

