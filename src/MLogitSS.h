#ifndef M_LOGIT_SS_H
#define M_LOGIT_SS_H

#include <RcppArmadillo.h>
struct MLogitSuffStats {
  arma::vec sum_Y;
  arma::vec sum_exp_lambda_minus_phi;

  MLogitSuffStats(int num_cat) {
    sum_Y                    = arma::zeros<arma::vec>(num_cat);
    sum_exp_lambda_minus_phi = sum_Y;
  }

  void Reset() {
    sum_Y                    = arma::zeros<arma::vec>(sum_Y.size());
    sum_exp_lambda_minus_phi = sum_Y;
  }

  void Increment(unsigned int j, const arma::vec& exp_lambda_minus, double phi) {
    sum_Y(j) = sum_Y(j) + 1;
    sum_exp_lambda_minus_phi =
      sum_exp_lambda_minus_phi + phi * exp_lambda_minus;
  }

};

#endif
