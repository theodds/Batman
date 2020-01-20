## Load ------------------------------------------------------------------------

library(Batman)
library(BART)
library(tidyverse)
library(mediation)
library(Matrix)
library(survival)

data(veteran)

## Fit -------------------------------------------------------------------------

X_surv_1 <- dplyr::select(veteran, trt, celltype, karno, diagtime, age, prior)
X_surv <- model.matrix(~ . - 1, data = X_surv_1)

args(surv.bart)
fit_bart <- surv.bart(X_surv, times = veteran$time, delta = veteran$status)


extract_idx <- function(iter_idx, trees, iter, num_tree) {
  start_idx <- (iter - 1) * num_tree + 1
  end_idx <- iter * num_tree + 1
  return(trees[iter_idx[start_idx]:(iter_idx[end_idx]-1),])
}

## Begin parsing ---------------------------------------------------------------

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

parsed_trees <- parse_trees(fit_bart, 1000, 50, 10)

out <- list()
for(i in 1:1000) {
  out[[i]] <- as.numeric(predict(parsed_trees[[i]],
                                 cbind(veteran$time, X_surv))[["yhat.test"]])
}

