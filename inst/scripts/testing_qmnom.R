## Load ------------------------------------------------------------------------

library(tidyverse)
library(Batman)
library(mvtnorm)
library(truncdist)
library(MCMCpack)
library(devtools)
library(dbarts)

## Generate data ---------------------------------------------------------------

N <- 1000
K <- 3
P <- 5

X     <- matrix(runif(N * P), nrow = N, ncol = P)
eta_1 <- 2 * X[,1] + X[,2]
eta_2 <- X[,1] + 4 * X[,3] * X[,2]
eta_3 <- X[,2] + 2 * X[,3]
eta   <- 3 * cbind(eta_1, eta_2, eta_3)

softmax <- function(x) exp(x) / sum(exp(x))
mu_0      <- apply(eta, 1, softmax) %>% t()
rho_0     <- 0.5
phi_0     <- 1 / (rho_0 + 1)
n         <- rep(1, N)

Y <- apply(mu_0, 1, \(m) rdirichlet(1, m * rho_0)) %>% t()

## Fitting the model -----------------------------------------------------------

probs          <- Matrix::Matrix(diag(P))
num_tree       <- 50
scale_lambda   <- 1 / sqrt(num_tree)
scale_lambda_0 <- 100
num_burn       <- 1000
num_thin       <- 1
num_save       <- 1000

fitted_qmnom <- QMultinomBart(X, Y, n, X, probs, num_tree, scale_lambda,
                              scale_lambda_0, num_burn, num_thin, num_save)

## Did we get phi correct? -----------------------------------------------------

hist(fitted_qmnom$phi)
abline(v = phi_0, lty = 3, lwd = 2)

## What about mu? --------------------------------------------------------------

mu_unnom <- exp(fitted_qmnom$lambda)
denoms   <- mu_unnom[,1,] + mu_unnom[,2,] + mu_unnom[,3,]
f <- function() {
  mu <- mu_unnom
  for(i in 1:dim(mu_unnom)[2]) {
    mu[,i,] <- mu[,i,] / denoms
  }
  return(mu)
}
mu <- f()
mu_hat <- apply(mu, c(1,2), mean)

par(mfrow=c(1,3))
plot(mu_hat[,1], mu_0[,1])
abline(a=0,b=1)
plot(mu_hat[,2], mu_0[,2])
abline(a=0,b=1)
plot(mu_hat[,3], mu_0[,3])
abline(a=0,b=1)
par(mfrow=c(1,1))

## What about relevant variables? ----------------------------------------------

plot(colMeans(fitted_qmnom$counts > 0), ylim = c(0,1))

## How does this compute with one-at-a-time? -----------------------------------

fitted_dbart <-
  bart(
    x.train = X,
    y.train = Y[, 1],
    x.test = X,
    ndpost = 4000,
    nskip = 4000
  )

## plot ------------------------------------------------------------------------

par(mfrow = c(1,2))
mu_hat_dbarts <- fitted_dbart$yhat.test.mean
plot(mu_hat_dbarts, mu_0[,1])
plot(mu_hat[,1], mu_0[,1], col = 'blue')


rmse <- function(x,y) sqrt(mean(abs(x - y)^2))
rmse(mu_hat_dbarts, mu_0[,1])
rmse(mu_hat[,1], mu_0[,1])

par(mfrow=c(1,1))
plot(density(fitted_dbart$yhat.test[,1]))
lines(density(mu[1,1,]))
abline(v = mu_0[1,1])

## Let's investigate the coverage of intervals and their widths! ---------------

mu_0_1    <- mu_0[,1]
mu_1      <- mu[,1,] %>% t()
mu_dbarts <- fitted_dbart$yhat.test

row_cis <- function(y) apply(y, 2, \(x) quantile(x, c(0.025, 0.975))) %>% t()
row_sds <- function(y) apply(y, 2, sd)

qmn_cis <- row_cis(mu_1)
bart_cis <- row_cis(mu_dbarts)
sd_qmn <- row_sds(mu_1)
sd_bart <- row_sds(mu_dbarts)

coverage_qmn  <- (mu_0_1 <= qmn_cis[,2] & mu_0_1 >= qmn_cis[,1])
coverage_bart <- (mu_0_1 <= bart_cis[,2] & mu_0_1 >= bart_cis[,1])

uq_data <-
  data.frame(
    sample = rep(1:N, 2),
    method = rep(c("QMN", "BART"), each = N),
    sds = c(sd_qmn, sd_bart),
    coverage = c(coverage_qmn, coverage_bart)
  )

uq_data %>% group_by(method) %>% summarise_all(mean)

## Junk having to do with doing the matrix inversion ---------------------------

## All it did was reduce to the usual X^2 statistics!!!

## # To reproduce
## set.seed(193487)

## mu <- runif(4)
## mu <- mu / sum(mu)
## mu <- mu[-4]
## D <- diag(mu)
## Dinv <- diag(1/mu)
## J <- matrix(1, nrow = 3, ncol = 3)
## y <- rnorm(3)

## ## Exact version
## solve(D - mu %*% t(mu))

## ## Woodburry
## Dinv + Dinv %*% mu %*% t(mu) %*% Dinv / as.numeric(1 - t(mu) %*% Dinv %*% mu)

## ## Simplified: verified!
## Dinv + J / (1 - sum(mu))

## ## What about inner product? Verified
## as.numeric(t(y) %*% (Dinv + J / (1 - sum(mu))) %*% y)
## sum(y^2 / mu) + sum(y)^2 / (1 - sum(mu))

## ## OK: does ordering matter? Under constraint that sum(y) = sum(mu) Verified!
## ## just need to remember that I need to subtract Y from mu, as it isn't true for
## ## arbitrary choices of y.

## set.seed(193487)

## mu <- runif(4)
## mu <- mu / sum(mu)
## y  <- runif(4)
## y  <-  y / sum(y)

## mu1 <- mu[-1]
## mu4 <- mu[-4]
## y1  <- y[-1] - mu1
## y4  <- y[-4] - mu4

## sum(y1^2 / mu1) + sum(y1)^2 / (mu[1])
## sum(y4^2 / mu4) + sum(y4)^2 / (mu[4])

## Sigma  <- diag(mu) - mu %*% t(mu)
## Sigma1 <- diag(mu1) - mu1 %*% t(mu1)
## Sigma4 <- diag(mu4) - mu4 %*% t(mu4)

## mvtnorm::dmvnorm(y1, sigma = Sigma1)
## mvtnorm::dmvnorm(y4, sigma = Sigma4)
## mvtnorm::dmvnorm(y, sigma = Sigma)

## ## What is going on? The maximum likelihood estimator of phi should not depend
## ## on which point I drop out? Is the issue that I'm not subtracting? I guess
## ## probably this is the issue

## ## Testing an alternate form ----

## sum((y - mu)^2 / (mu * (1 - mu)))
## sum((y - mu)^2 / mu)

## ## What about stationary distribution? ----

## ## Let's look at a simple setting of the Quasi-Poisson under the
## ## pseudo-likelihood approach. Under this, we have lambda ~ Gam(S/phi, N/phi)
## ## and phi ~ Gam(N/2, SSE/2). What happens to the chain?

## lambda_0 <- 2
## N        <- 500
## phi_0    <- 2
## Y        <- phi_0 * rpois(n = N, lambda = lambda_0 / phi_0)

## num_warmup  <- 0
## num_save    <- 5000
## num_iter    <- num_save + num_warmup
## lambda      <- 2
## phi         <- 2
## lambda_save <- numeric(num_save)
## phi_save    <- numeric(num_save)
## idx         <- 1

## for(i in 1:num_iter) {
##   phi <- 1 / rtrunc(1, spec = "gamma", a = 1/10, b = Inf,
##                     shape = N/2, rate = sum((Y - lambda)^2) / 2 / lambda)
##   lambda <- rtrunc(1, spec = "gamma", a = 1/10, b = Inf,
##                    shape = sum(Y) / phi,
##                    rate = N / phi)
##   ## rgamma(1, sum(Y) / phi, N / phi)

##   if(i > num_warmup) {
##     lambda_save[idx] <- lambda
##     phi_save[idx] <- phi
##     idx <- idx + 1
##   }
## }

## par(mfrow = c(1,2))
## plot(phi_save, type = 'l')
## plot(lambda_save, type = 'l')

## mean(phi_save)
## mean(lambda_save)

## ggplot() + geom_density_2d(aes(x = phi_save, y = lambda_save))

## ## Is it maybe an ordering thing? ----

## par(mfrow = c(1,2))

## # phi then labmda
## plot(phi_save, lambda_save)

## # lambda, then phi
## plot(phi_save[-1], lambda_save[-num_save])
