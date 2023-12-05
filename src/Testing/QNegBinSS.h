#ifndef QNB_SS_H
#define QNB_SS_H

#include <RcppArmadillo.h>

struct QNBSuffStats {
  double sum_Y;
  double sum_Y_lambda_minus;
  double sum_exp_lambda_minus;

  QNBSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Y                = 0.0;
    sum_Y_lambda_minus   = 0.0;
    sum_exp_lambda_minus = 0.0;
  }

  void Increment(double y, double lambda_minus, double phi, double xi) {
    sum_Y                += y / phi;
    sum_exp_lambda_minus += xi * exp(lambda_minus) / phi;
  }

};

#endif
