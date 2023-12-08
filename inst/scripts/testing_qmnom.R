## Load ----

library(tidyverse)
library(Batman)
library(mvtnorm)

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
