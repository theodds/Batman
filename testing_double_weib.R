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