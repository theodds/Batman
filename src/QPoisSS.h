#ifndef QPOIS_SS_H
#define QPOIS_SS_H

#include <RcppArmadillo.h>

struct QPoisSuffStats {
  double sum_Y;
  double sum_Y_lambda_minus;
  double sum_exp_lambda_minus;

  QPoisSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Y                = 0.0;
    sum_Y_lambda_minus   = 0.0;
    sum_exp_lambda_minus = 0.0;
  }

  void Increment(double y, double lambda_minus, double phi) {
    sum_Y                += y / phi;
    sum_Y_lambda_minus   += y * lambda_minus / phi;
    sum_exp_lambda_minus += exp(lambda_minus) / phi;
  }

};

#endif
