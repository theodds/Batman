#ifndef QBINOM_SS_H
#define QBINOM_SS_H

#include <RcppArmadillo.h>

struct QBinomSuffStats {
  double sum_Y_by_phi;
  double sum_exp_lambda_minus_by_phi;

  QBinomSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Y_by_phi                = 0.0;
    sum_exp_lambda_minus_by_phi = 0.0;
  }

  void Increment(double y, double rho, double lambda_minus, double phi) {
    sum_Y_by_phi                += y / phi;
    sum_exp_lambda_minus_by_phi += rho * exp(lambda_minus) / phi;
  }

};

#endif
