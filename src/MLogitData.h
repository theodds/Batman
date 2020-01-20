#ifndef M_LOGIT_DATA_H
#define M_LOGIT_DATA_H

#include <RcppArmadillo.h>

struct MLogitData {
  arma::mat  X;
  arma::uvec Y;
  arma::mat  lambda_hat;
  arma::vec  phi;
  int num_class;

MLogitData(const arma::mat& X_, const arma::uvec& Y_, int num_class_) :
  num_class(num_class_)
  {
    X          = X_;
    Y          = Y_;
    phi        = arma::ones<arma::vec>(Y.size()); // TODO better initialize? Update first?
    lambda_hat = arma::zeros<arma::mat>(Y.size(), num_class);
  }
};

#endif
