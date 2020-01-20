#ifndef POIS_SS_H
#define POIS_SS_H

#include <RcppArmadillo.h>

struct PoisSuffStats {
  double sum_Y;
  double sum_Y_lambda_minus;
  double sum_exp_lambda_minus;

  PoisSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Y                = 0.0;
    sum_Y_lambda_minus   = 0.0;
    sum_exp_lambda_minus = 0.0;
  }

  void Increment(double y, double lambda_minus) {
    sum_Y                += y;
    sum_Y_lambda_minus   += y * lambda_minus;
    sum_exp_lambda_minus += exp(lambda_minus);
  }

};

#endif
