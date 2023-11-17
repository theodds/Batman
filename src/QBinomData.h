#ifndef QBINOM_DATA_H
#define QBINOM_DATA_H

#include <RcppArmadillo.h>

struct QBinomData {
  arma::mat X;
  arma::vec Y;
  arma::vec n;
  arma::vec lambda_hat;
  arma::vec phi;

  QBinomData(const arma::mat& X_, const arma::vec& Y_, const arma::vec& n_) {
    X = X_;
    Y = Y_;
    n = n_;
    lambda_hat = arma::zeros<arma::vec>(X.n_rows);
    phi = arma::ones<arma::vec>(X.n_rows);
  }

};

#endif

