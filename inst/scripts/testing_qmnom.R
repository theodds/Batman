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

## OK: does ordering matter? Under constraint that sum(y) = sum(mu)

set.seed(193487)

mu <- runif(4)
mu <- mu / sum(mu)
y  <- runif(4)
y  <-  y / sum(y)

y1  <- y[-1]
mu1 <- mu[-1]
y4  <- y[-4]
mu4 <- mu[-4]

sum(y1^2 / mu1) + sum(y1)^2 / (1 - sum(mu1))
sum(y4^2 / mu4) + sum(y4)^2 / (1 - sum(mu4))

Sigma  <- diag(mu) - mu %*% t(mu)
Sigma1 <- diag(mu1) - mu1 %*% t(mu1)
Sigma4 <- diag(mu4) - mu4 %*% t(mu4)

mvtnorm::dmvnorm(y1, sigma = Sigma1)
mvtnorm::dmvnorm(y4, sigma = Sigma4)
mvtnorm::dmvnorm(y, sigma = Sigma)
