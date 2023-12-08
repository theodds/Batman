#ifndef QMNOM_DATA_H
#define QMNOM_DATA_H

#include <RcppArmadillo.h>

struct QMultinomData {
  arma::mat X;
  arma::mat Y;
  arma::vec n;
  arma::mat lambda_hat;
  arma::vec rho;

  QMultinomData(const arma::mat& X_, const arma::mat& Y_, const arma::vec& n_) {
    X = X_;
    Y = Y_;
    n = n_;
    lambda_hat = arma::zeros<arma::mat>(Y.n_rows, Y.n_cols);
    rho = arma::ones<arma::vec>(Y.n_rows);
  }
};

#endif

