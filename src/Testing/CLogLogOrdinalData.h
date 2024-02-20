#ifndef CLOGLOGORDINAL_DATA_H
#define CLOGLOGORDINAL_DATA_H

#include <RcppArmadillo.h>

struct CLogLogOrdinalData {
  arma::mat X;
  arma::uvec Y;
  arma::vec Z;
  arma::vec lambda_hat;

  CLogLogOrdinalData(const arma::mat& X_, const arma::vec& Y_) {
    X = X_;
    Y = Y_;
    Z = arma::ones<arma::vec>(X.n_rows);
    lambda_hat = arma::zeros<arma::vec>(X.n_rows);
  }

};

#endif

