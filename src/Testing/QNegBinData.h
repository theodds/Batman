#ifndef QPOIS_DATA_H
#define QPOIS_DATA_H

#include <RcppArmadillo.h>

struct QPoisData {
  arma::mat X;
  arma::vec Y;
  arma::vec lambda_hat;

  QPoisData(const arma::mat& X_, const arma::vec& Y_) {
    X = X_;
    Y = Y_;
    lambda_hat = arma::zeros<arma::vec>(X.n_rows);
  }

};

#endif

