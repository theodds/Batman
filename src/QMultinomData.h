#ifndef QMNOM_DATA_H
#define QMNOM_DATA_H

#include <RcppArmadillo.h>

struct QMultinomData {
  arma::mat X;
  arma::mat Y;
  arma::mat Z;
  arma::vec n;
  arma::mat lambda_hat;
  arma::vec rho;

  QMultinomData(const arma::mat& X_, const arma::mat& Y_, const arma::vec& n_) {
    X = X_;
    Y = Y_;
    n = n_;
    Z = Y;
    for(int i = 0; i < Y.n_rows; i++) {
      Z.row(i) = Z.row(i) * n(i);
    }
    lambda_hat = arma::zeros<arma::mat>(Y.n_rows, Y.n_cols);
    rho = arma::ones<arma::vec>(Y.n_rows);
  }
};

#endif

