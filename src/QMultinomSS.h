#ifndef QMNOM_SS_H
#define QMNOM_SS_H

#include <RcppArmadillo.h>

struct QMultinomSuffStats {
  arma::vec sum_Z_by_phi;
  arma::vec sum_exp_lambda_minus_by_phi;

  QMultinomSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Z_by_phi                = arma::zeros<arma::vec>(sum_Z_by_phi.n_elem);
    sum_exp_lambda_minus_by_phi = arma::zeros<arma::vec>(sum_Z_by_phi.n_elem);
  }

  void Increment(arma::vec z, double rho,
                 arma::vec lambda_minus, double phi) {
    sum_Z_by_phi                += z / phi;
    sum_exp_lambda_minus_by_phi += rho * exp(lambda_minus);
  }
};

#endif
