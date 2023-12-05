#ifndef QNB_DATA_H
#define QNB_DATA_H

#include <RcppArmadillo.h>

struct QNBData {
  arma::mat X;
  arma::vec Y;
  arma::vec lambda_hat;
  arma::vec xi;

  QNBData(const arma::mat& X_, const arma::vec& Y_) {
    X = X_;
    Y = Y_;
    lambda_hat = arma::zeros<arma::vec>(X.n_rows);
    xi = arma::ones<arma::vec>(X.n_rows);
  }

};

#endif

