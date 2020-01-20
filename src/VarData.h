#ifndef VAR_DATA_H
#define VAR_DATA_H

#include <RcppArmadillo.h>

struct VarData {
  arma::mat X;
  arma::vec Y;
  arma::mat mu_hat;
  arma::mat tau_hat;

VarData(const arma::mat& X_, const arma::vec& Y_) : X(X_), Y(Y_)
  {
    mu_hat  = arma::zeros<arma::vec>(Y.size());
    tau_hat = arma::ones<arma::vec>(Y.size());
  }
};

#endif
