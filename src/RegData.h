#ifndef REG_DATA_H
#define REG_DATA_H

#include <RcppArmadillo.h>


struct RegData {
  arma::mat X;
  arma::vec Y;
  arma::vec mu_hat;

  RegData(const arma::mat& X_, const arma::vec& Y_) {
    X = X_;
    Y = Y_;
    mu_hat = arma::zeros<arma::vec>(X.n_rows);
  }
};




#endif
