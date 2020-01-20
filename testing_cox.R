# Load --------------------------------------------------------------------

library(survival)
library(Batman)

# Load and format data ----------------------------------------------------

data(veteran)

head(veteran)

Y     <- veteran$time

{
  veteran_X <- cbind(veteran$trt, veteran$karno, veteran$diagtime, 
                     veteran$age, veteran$prior)  
  veteran_X[,1] <- veteran_X[,1] - 1
  veteran_X[,2] <- veteran_X[,2] / 100
  veteran_X[,3] <- veteran_X[,3] / 90
  veteran_X[,4] <- (veteran_X[,4] - 34) / (81 - 34)
  veteran_X[,5] <- veteran_X[,5] / 10
}


delta <- veteran$status
o     <- order(Y) - 1
gaps  <- o - 1
probs <- Matrix::Matrix(diag(5), sparse = TRUE)
num_trees <- 30
scale_lambda <- 1 / sqrt(num_trees)
num_burn <- 4000
num_thin <- 1
num_save <- 4000


# Process survival data ---------------------------------------------------

process_surv <- function(Y, delta, X) 
{
  o     <- order(Y, 1-delta)
  Y     <- Y[o]
  delta <- delta[o]
  X     <- X[o,]
  survs <- Surv(Y, delta)
  k <- 1
  U <- numeric(length(unique(survs)))
  L <- numeric(length(unique(survs)))
  L[1] <- 1
  for(i in 2:length(survs))
  {
    if(!identical(survs[i], survs[i-1])) {
      U[k] <- i - 1
      k <- k + 1
      L[k] <- i
    }
  }
  U[length(unique(survs))] <- length(survs)
  return(list(o = 1:length(Y) - 1, Y = Y, L = L-1, U = U-1, delta = delta, X = X))
}

proc_dat <- process_surv(Y, delta, veteran_X)

# Fit ---------------------------------------------------------------------

set.seed(816740391)

fit_coxbart <- CoxBart(proc_dat$X, 
                       proc_dat$Y, 
                       proc_dat$delta, 
                       proc_dat$o, 
                       proc_dat$L, 
                       proc_dat$U, 
                       probs, 
                       num_trees, scale_lambda, num_burn, num_thin, num_save)


# Goodness of fit ---------------------------------------------------------

plot(rank(proc_dat$Y), colMeans(fit_coxbart$lambda))

ham <- Surv(time = proc_dat$Y, event = proc_dat$delta, type = "right")
concordance(ham ~ colMeans(-fit_coxbart$lambda))

