#ifndef GAMMA_SS_H
#define GAMMA_SS_H

#include <RcppArmadillo.h>

struct GammaSuffStats {
  double sum_1_by_phi;
  double sum_exp_lambda_minus_y;

  GammaSuffStats() {
    Reset();
  }

  void Reset() {
    sum_1_by_phi = 0.;
    sum_exp_lambda_minus_y = 0.;
  }

  void Increment(double y, double lambda_minus, double phi) {
    sum_1_by_phi            += 1. / phi;
    sum_exp_lambda_minus_y  += exp(lambda_minus) * y / phi;
  }

};

#endif
