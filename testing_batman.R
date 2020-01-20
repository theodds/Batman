# Load --------------------------------------------------------------------

library(Batman)
library(tidyverse)
library(Matrix)
library(zeallot)
library(DPpackage)


# Misc funcs --------------------------------------------------------------

rlines <- function(x,y, ...) {
  o <- order(x)
  lines(x[o], y[o], ...)
}

true_dens <- function(y,x) exp(-2 * x) * dnorm(y,x,0.1) + (1 - exp(-2*x)) * dnorm(y,x^4,0,.2)

# Generate data -----------------------------------------------------------

set.seed(32078)
gen_data <- function(n, p) {
  X <- matrix(runif(n * p), nrow = n)
  p <- exp(-2 * X[,2])
  Y <- ifelse(
    test = runif(n) < p, 
    yes = rnorm(n, X[,2], sqrt(.01)), 
    no = rnorm(n, X[,2]^4, sqrt(.04))
    )
  mu <- p * X[,2] + (1 - p) * X[,2]^4
  return(list(X = X, Y = Y, mu = mu))
}

c(X,Y,mu) %<-% gen_data(n = 500, p = 2)

# Fit model ---------------------------------------------------------------

set.seed(507)
probs <- Matrix(diag(ncol(X)), sparse = TRUE)
# probs[1,1] <- 0

num_tree <- 50
num_cat <- 10
X_test <- X[1:5, ]
X_test[1:5,2] <- c(0.1, 0.25, 0.49, 0.75, 0.88)
Y_test <- seq(from = min(Y), to = max(Y), length = 200)

out <- Batman(X = X, 
       Y = Y, 
       probs = probs, 
       num_cat = num_cat, 
       num_tree = num_tree, 
       scale_lambda = num_cat/sqrt(num_tree), 
       shape_lambda_0 = 1/num_cat, 
       rate_lambda_0 = 1, 
       scale_kappa = 1/sqrt(num_tree), 
       sigma_scale_log_tau = .1/sqrt(num_tree), 
       shape_tau_0 = 1, 
       rate_tau_0 = 1, 
       num_burn = 4000, 
       num_thin = 5, 
       num_save = 4000, 
       X_test = X_test, 
       Y_test = Y_test)


# hist(out$class[4000,])


# plot(X[,2], 1/sqrt(out$tau_hat))

# junk --------------------------------------------------------------------


dim(X)

tmpf <- function(i) {
  x <- X_test[i,2]
  plot(function(y) {
    p <- exp(-2 * x)
    p * dnorm(y, x, 0.1) + (1 - p) * dnorm(y, x^4, 0.2)
  }, xlim = c(-0.5,1.5), ylim = c(0,4), col = 'green')
  
  lines(Y_test, rowMeans(exp(out$dens[i,,])))
  # lcl <- apply(exp(out$dens[i,,]), 1, function(x) quantile(x,0.025))
  # ucl <- apply(exp(out$dens[i,,]), 1, function(x) quantile(x,0.975))
  my_intervals <- apply(exp(out$dens[i,,]), 1, function(x) hdi(x, .95))
  lcl <- my_intervals[1,]
  ucl <- my_intervals[2,]
  
  lines(Y_test, lcl, col = 'skyblue2')
  lines(Y_test, ucl, col = 'skyblue2')
}

par(mfrow = c(2,3))
sapply(1:5, tmpf)

plot(X[,2], out$mu_hat)
rlines(X[,2], mu)
abline(a=0,b=1)
plot(function(x) x^4, add = TRUE)
# plot(X[,2], 1/sqrt(out$tau_hat))



# dpcdensity --------------------------------------------------------------

# Prior information
w <- cbind(Y,X[,2])
wbar <- apply(w,2,mean)
wcov <- var(w)
prior <- list(a0=10,
              b0=1,
              nu1=4,
              nu2=4,
              s2=0.5*wcov,
              m2=wbar,
              psiinv2=2*solve(wcov),
              tau1=6.01,
              tau2=2.01)
# Initial state
state <- NULL
# MCMC parameters
mcmc <- list(nburn=5000,
             nsave=5000,
             nskip=3,
             ndisplay=100)
# fitting the model
xpred <- c(0.1, 0.25, 0.49, 0.75, 0.88)

fit <- DPcdensity(y=Y,x=X[,2],xpred=xpred,ngrid=100,
                  prior=prior,
                  mcmc=mcmc,
                  state=state,
                  status=TRUE,
                  compute.band=TRUE,type.band="PD")


# Plot results ------------------------------------------------------------

tmpf2 <- function(i) {
  plot(function(y) {
    x <- X_test[i,2]
    p <- exp(-2 * x)
    p * dnorm(y, x, 0.1) + (1 - p) * dnorm(y, x^4, 0.2)
  }, xlim = c(-0.5,1.5), ylim = c(0,4), col = 'green')
  lines(fit$grid, fit$densp.h[i,], lwd = 1, lty = 2, 
       main = paste0("x = ", X_test[i,2]), xlab = "values", ylab  = "density", 
       ylim = c(0,4))
  lines(fit$grid, fit$densp.l[i,], lwd = 1, lty = 2)  
  lines(fit$grid, fit$densp.m[i,], lwd = 1, lty = 1)
}

sapply(1:5, tmpf2)
