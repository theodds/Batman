#ifndef RVAR_DATA_H
#define RVAR_DATA_H

#include <RcppArmadillo.h>

struct RVarData {
  arma::mat X;
  arma::vec Y;
  arma::vec tau_hat;

RVarData(const arma::mat& X_, const arma::vec& Y_) : X(X_), Y(Y_)
  {
    tau_hat = arma::ones<arma::vec>(Y.size());
  }
};

#endif
