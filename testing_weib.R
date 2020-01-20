## TODO: Add draw_prior to package
## TODO: Make Github Repo

## Load Package ----

library(Batman)
library(Rcpp)
library(tidyverse)
library(Matrix)

set.seed(1121)

## Generate data ----

generate_data <- function(N, P, sigma, weibull_power) {
  X <- matrix(rnorm(N*P), nrow = N)
  true_rate <- exp(sigma * X[,1])
  true_scale <- true_rate^(-1/weibull_power)
  Y <- rweibull(N, shape = weibull_power, true_scale)
  
  out <- list(X = X, 
              Y = Y, 
              true_rate = true_rate, 
              true_scale = true_scale,
              weibull_power = weibull_power)
}

my_data <- generate_data(N = 1000, P = 3, sigma = .5, weibull_power = 1)
my_data$qX <- SoftBart::quantile_normalize_bart(my_data$X)

plot(my_data$Y, my_data$true_rate)

## Default params ----

my_data$W <- my_data$Y
my_data$idx <- 1:length(my_data$Y) - 1
my_data$probs <- Matrix::Matrix(diag(ncol(my_data$X)), sparse = TRUE)
my_data$num_trees <- 50
my_data$scale_lambda <- 1.5 * sd(log(my_data$Y)) / sqrt(my_data$num_trees) 
my_data$shape_lambda_0 <- 1
my_data$rate_lambda_0 <- 1
my_data$num_burn <- 5000
my_data$num_thin <- 1
my_data$num_save <- 5000
my_data$do_ard <- TRUE
my_data$update_alpha <- TRUE


## Make Weib ----

weib_forest <- with(my_data, MakeWeib(probs = probs, 
                                      num_trees = as.integer(num_trees), 
                                      scale_lambda = scale_lambda, 
                                      shape_lambda_0 = shape_lambda_0, 
                                      rate_lambda_0 = rate_lambda_0, 
                                      weibull_power = weibull_power))

burned_samples <- with(my_data, weib_forest$do_gibbs(qX, Y, W, idx, qX, 500))
weib_forest$do_ard()
burned_samples <- with(my_data, weib_forest$do_gibbs(qX, Y, W, idx, qX, 500))
saved_samples <- with(my_data, weib_forest$do_gibbs(qX, Y, W, idx, qX, 1000))

plot(1/my_data$true_rate, colMeans(exp(-saved_samples)))
abline(a=0,b=1)

## Run Weibull ----

out <- with(my_data,WeibBart(qX, Y, W, idx, probs, num_trees, scale_lambda, 
                             shape_lambda_0, rate_lambda_0, weibull_power,
                             do_ard, update_alpha,
                             num_burn, num_thin, num_save))


## Plot? ----

plot(colMeans(out$lambda), log(my_data$true_rate))
abline(a=0,b=1)



