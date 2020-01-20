## Load ------------------------------------------------------------------------

library(Batman)
library(BART)
library(tidyverse)
library(mediation)
library(Matrix)
library(mitools)
data(jobs)


# Function for building probabilities out of predictors -------------------

MakeProbs <- function(groups) {
  probs <- Matrix(0, nrow = length(groups), ncol = max(groups), sparse = TRUE)
  num_group <- max(groups)
  for(i in 1:num_group) {
    probs[which(groups == i),i] <- 1
    probs[which(groups == i),i] <- 1 / sum(probs[which(groups==i),i])
  }
  return(probs)
}

## Function for extracting iterations of trees ---------------------------------

parse_trees <- function(fit, n_iter, n_tree, n_pred) {
  out <- list()
  tc <- textConnection(fit$treedraws$trees)
  trees <- read.table(file = tc, fill = TRUE, row.names = NULL, header = FALSE,
                      col.names = c("node", "var", "cut", "leaf"))
  iter_idx <- which(is.na(trees$var))

  tc <- textConnection(fit$treedraws$trees)
  base_str <- paste(1, n_tree, n_pred)
  readLines(tc, 1)

  for(i in 1:n_iter) {
    start  <- n_tree * (i - 1) + 1 
    end    <- n_tree * i + 1
    if(i == n_iter) {
      tree_str_1 <- readLines(tc)
      tree_str_2 <- paste(c(base_str, tree_str_1), collapse = "\n")
      tree_str_3 <- tree_str_2
    } else {
      tree_str_1 <- readLines(tc, iter_idx[end] - iter_idx[start])
      tree_str_2 <- paste(c(base_str, tree_str_1), collapse = "\n")
      tree_str_3 <- paste0(tree_str_2, "\n", collapse = "")
    }
    out[[i]]   <- fit
    out[[i]]$treedraws$trees <- tree_str_3
  }
  return(out)
}


## Format predictors ----------------------------------------------------------

X_y_1 <- jobs %>% dplyr::select(sex, age, marital, 
                                nonwhite, educ, income, job_seek)
X_y_2 <- X_y_1 %>% mutate(nonwhite = ifelse(nonwhite == "white0", 0, 1), 
                          educ = as.numeric(educ), income = as.numeric(income))
X_y_3 <- model.matrix(~ . - 1, data = X_y_2)

X_y <- SoftBart::quantile_normalize_bart(X_y_3)
colnames(X_y) <- colnames(X_y_3)
X_m <- as.matrix(dplyr::select(as.data.frame(X_y), -job_seek))

Y   <- jobs$depress2
M   <- jobs$job_seek

Y_surv           <- round(11 * Y) / 11
M_surv           <- round(12 * M) / 12 / 5
X_y[,"job_seek"] <- M_surv

trt <- jobs$treat

groups_y <- attributes(X_y_3)[["assign"]]
probs_y  <- MakeProbs(groups_y)

GROUP_SEEK <- 7
IDX_SEEK   <- 11

probs_m <- probs_y[-IDX_SEEK, -GROUP_SEEK]



## Function to fit models  -----------------------------------------------------

stick_break <- function(x) {
  t <- length(x)
  cs <- cumsum(log(1 - x))
  p <- log(x) + c(0, cs[1:(t-1)])
  return(exp(p))
}

fit_med_ord <- function(t, trt, Y, X, ndpost = 1000, 
                        nskip = 250, keepevery = 10, ...) {
  
  Y_t <- Y[trt == t]
  X_t <- X[trt == t, ]
  delta <- 0 * Y_t + 1
  
  unique_Y <- unique(Y_t)
  num_unique <- length(unique_Y)
  num_obs    <- nrow(X)
  
  surv_pre <- surv.pre.bart(times = Y_t, delta = delta, 
                            x.train = X_t, x.test = X)
  
  fit <- surv.bart(x.train = surv_pre$tx.train, 
                   y.train = surv_pre$y.train, 
                   x.test = surv_pre$tx.test, 
                   ndpost = ndpost, 
                   nskip = nskip, 
                   keepevery = keepevery,
                   ...)
  
  phat <- array(pnorm(fit$yhat.test), c(ndpost, num_unique, num_obs))
  phat[,num_unique,] <- 1
  

  
  qhat <- apply(phat, c(1,3), stick_break)
  
  return(list(fit = fit, delta = delta, qhat = qhat, Y_vals = sort(unique_Y)))
  
}


## Fit Models ------------------------------------------------------------------

set.seed(840984)
fit_m_1 <- fit_med_ord(t = 1, trt = trt, Y = M_surv, X = X_m)
set.seed(840984+1)
fit_m_0 <- fit_med_ord(t = 0, trt = trt, Y = M_surv, X = X_m)
set.seed(840984+2)
fit_y_1 <- fit_med_ord(t = 1, trt = trt, Y = Y_surv, X = X_y)
set.seed(840984+3)
fit_y_0 <- fit_med_ord(t = 0, trt = trt, Y = Y_surv, X = X_y)

n_iter <- nrow(fit_m_1$fit$yhat.test)
n_tree <- 50
treedraws_m_1 <- parse_trees(fit = fit_m_1$fit, 
                             n_iter = n_iter, 
                             n_tree = n_tree, 
                             n_pred = ncol(X_m) + 1)
treedraws_m_0 <- parse_trees(fit = fit_m_0$fit, 
                             n_iter = n_iter, 
                             n_tree = n_tree, 
                             n_pred = ncol(X_m) + 1)
treedraws_y_1 <- parse_trees(fit = fit_y_1$fit, 
                             n_iter = n_iter, 
                             n_tree = n_tree, 
                             n_pred = ncol(X_y) + 1)
treedraws_y_0 <- parse_trees(fit = fit_y_0$fit, 
                             n_iter = n_iter, 
                             n_tree = n_tree, 
                             n_pred = ncol(X_y) + 1)

## G-Computation ---------------------------------------------------------------

library(MCMCpack)
library(parallel)

g_comp <- function(fit_y, Y, fit_m, M, X_m, omega, num_iter, num_impute, num_g) {
  N <- nrow(X_m)
  
  ## unique values
  y_unique <- sort(unique(Y))
  m_unique <- sort(unique(M))
  num_y    <- length(y_unique)
  num_m    <- length(m_unique)
  
  ## Output for mu and sigma
  samp_mu    <- numeric(num_impute)
  samp_sigma <- numeric(num_impute)
  iters      <- floor(round(seq(from = 1, to = num_iter, length = num_impute)))
  
  for(i in 1:num_impute) {
    it <- iters[i]
    
    ## sample x
    rand_x <- sample(1:N, size = num_g, prob = omega[it,], replace = TRUE)
    
    ## sample m
    x_m_1           <- do.call(rbind, lapply(1:num_m, function(i) X_m[rand_x,]))
    x_m             <- cbind(rep(m_unique, each = num_g), x_m_1)
    pred_m          <- predict(fit_m[[it]], x_m)
    phat_m          <- array(pnorm(pred_m$yhat.test), c(1, num_g, num_m))
    phat_m[,,num_m] <- 1
    qhat_m          <- apply(phat_m, c(1,2), stick_break)
    m_samp          <- sapply(1:num_g, function(i) sample(m_unique, 1, prob = qhat_m[,1,i]))
    
    ## compute the mean
    xy_1 <- cbind(X_m[rand_x,], m_samp)
    xy_2 <- do.call(rbind, lapply(1:num_y, function(i) xy_1))
    xy_3 <- cbind(rep(y_unique, each = num_g), xy_2)
    xy   <- xy_3
    pred_x <- predict(fit_y[[it]], xy)
    phat_x <- array(pnorm(pred_x$yhat.test), c(1, num_g, num_y))
    phat_x[,,num_y] <- 1
    qhat_x <- apply(phat_x, c(1,2), stick_break)
    
    samps <- sapply(1:num_g, function(i) sum(qhat_x[,1,i] * y_unique))
    
    samp_mu[i] <- mean(samps)
    samp_sigma[i] <- sd(samps) / sqrt(num_g)
    cat("\rFinishing iteration ", i, "\t\t\t")
    
  }
  
  out <- data.frame(estimate = samp_mu, standard_error = samp_sigma)
  return(out)
  
}


## Do G-comp -------------------------------------------------------------------

## Simulate x
set.seed(840984+8)
omega <- rdirichlet(n = nrow(fit_y_1$fit$varcount), alpha = rep(1,nrow(X_m)))

set.seed(840984+4)
eta_1_1 <- g_comp(treedraws_y_1, Y_surv[trt == 1], 
                  treedraws_m_1, M_surv[trt == 1], 
                  X_m, omega, 1000, 100, 500)

set.seed(840984+5)
eta_1_0 <- g_comp(treedraws_y_1, Y_surv[trt == 1], 
                  treedraws_m_0, M_surv[trt == 0], 
                  X_m, omega, 1000, 100, 500)

set.seed(840984+6)
eta_0_1 <- g_comp(treedraws_y_0, Y_surv[trt == 0], 
                  treedraws_m_1, M_surv[trt == 1], 
                  X_m, omega, 1000, 100, 500)

set.seed(840984+7)
eta_0_0 <- g_comp(treedraws_y_0, Y_surv[trt == 0], 
                  treedraws_m_0, M_surv[trt == 0], 
                  X_m, omega, 1000, 100, 500)

my_combine <- function(x) MIcombine(as.list(x$estimate), 
                                    as.list(x$standard_error^2))

delta_1 <- data.frame(estimate = eta_1_1$estimate - eta_0_1$estimate, 
                      standard_error = sqrt(eta_1_1$standard_error^2 + eta_0_1$standard_error^2))

e11r <- my_combine(eta_1_1)
e10r <- my_combine(eta_1_0)
e01r <- my_combine(eta_0_1)
e00r <- my_combine(eta_0_0)
