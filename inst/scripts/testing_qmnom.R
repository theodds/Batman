## Load ----

library(tidyverse)
library(Batman)
library(mvtnorm)
library(truncdist)

## Testing Woodburry ----

# To reproduce
set.seed(193487)

mu <- runif(4)
mu <- mu / sum(mu)
mu <- mu[-4]
D <- diag(mu)
Dinv <- diag(1/mu)
J <- matrix(1, nrow = 3, ncol = 3)
y <- rnorm(3)

                                        # Exact version
solve(D - mu %*% t(mu))

                                        # Woodburry
Dinv + Dinv %*% mu %*% t(mu) %*% Dinv / as.numeric(1 - t(mu) %*% Dinv %*% mu)

                                        # Simplified: verified!
Dinv + J / (1 - sum(mu))

                                        # What about inner product? Verified
as.numeric(t(y) %*% (Dinv + J / (1 - sum(mu))) %*% y)
sum(y^2 / mu) + sum(y)^2 / (1 - sum(mu))

## OK: does ordering matter? Under constraint that sum(y) = sum(mu) Verified!
## just need to remember that I need to subtract Y from mu, as it isn't true for
## arbitrary choices of y.

set.seed(193487)

mu <- runif(4)
mu <- mu / sum(mu)
y  <- runif(4)
y  <-  y / sum(y)

mu1 <- mu[-1]
mu4 <- mu[-4]
y1  <- y[-1] - mu1
y4  <- y[-4] - mu4

sum(y1^2 / mu1) + sum(y1)^2 / (mu[1])
sum(y4^2 / mu4) + sum(y4)^2 / (mu[4])

Sigma  <- diag(mu) - mu %*% t(mu)
Sigma1 <- diag(mu1) - mu1 %*% t(mu1)
Sigma4 <- diag(mu4) - mu4 %*% t(mu4)

mvtnorm::dmvnorm(y1, sigma = Sigma1)
mvtnorm::dmvnorm(y4, sigma = Sigma4)
mvtnorm::dmvnorm(y, sigma = Sigma)

## What is going on? The maximum likelihood estimator of phi should not depend
## on which point I drop out? Is the issue that I'm not subtracting? I guess
## probably this is the issue

## Testing an alternate form ----

sum((y - mu)^2 / (mu * (1 - mu)))
sum((y - mu)^2 / mu)

## What about stationary distribution? ----

## Let's look at a simple setting of the Quasi-Poisson under the
## pseudo-likelihood approach. Under this, we have lambda ~ Gam(S/phi, N/phi)
## and phi ~ Gam(N/2, SSE/2). What happens to the chain?

lambda_0 <- 2
N        <- 500
phi_0    <- 2
Y        <- phi_0 * rpois(n = N, lambda = lambda_0 / phi_0)

num_warmup  <- 0
num_save    <- 5000
num_iter    <- num_save + num_warmup
lambda      <- 2
phi         <- 2
lambda_save <- numeric(num_save)
phi_save    <- numeric(num_save)
idx         <- 1

for(i in 1:num_iter) {
  phi <- 1 / rtrunc(1, spec = "gamma", a = 1/10, b = Inf,
                    shape = N/2, rate = sum((Y - lambda)^2) / 2 / lambda)
  lambda <- rtrunc(1, spec = "gamma", a = 1/10, b = Inf,
                   shape = sum(Y) / phi,
                   rate = N / phi)
  ## rgamma(1, sum(Y) / phi, N / phi)

  if(i > num_warmup) {
    lambda_save[idx] <- lambda
    phi_save[idx] <- phi
    idx <- idx + 1
  }
}

par(mfrow = c(1,2))
plot(phi_save, type = 'l')
plot(lambda_save, type = 'l')

mean(phi_save)
mean(lambda_save)

ggplot() + geom_density_2d(aes(x = phi_save, y = lambda_save))

## Is it maybe an ordering thing? ----

par(mfrow = c(1,2))

# phi then labmda
plot(phi_save, lambda_save)

# lambda, then phi
plot(phi_save[-1], lambda_save[-num_save])
