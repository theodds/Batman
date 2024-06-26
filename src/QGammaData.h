#ifndef QGAMMA_DATA_H
#define QGAMMA_DATA_H

#include <RcppArmadillo.h>

struct QGammaData {
  arma::mat X;
  arma::vec Y;
  arma::vec lambda_hat;

  QGammaData(const arma::mat& X_, const arma::vec& Y_) {
    X = X_;
    Y = Y_;
    lambda_hat = arma::zeros<arma::vec>(X.n_rows);
  }

};

#endif

