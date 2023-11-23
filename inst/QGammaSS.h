#ifndef QGAMMA_SS_H
#define QGAMMA_SS_H

#include <RcppArmadillo.h>

struct QGammaSuffStats {
  double sum_lambda_minus;
  double sum_exp_lambda_minus_y;

  QGammaSuffStats() {
    Reset();
  }

  void Reset() {
    sum_lambda_minus = 0.;
    sum_exp_lambda_minus_y = 0.;
  }

  void Increment(double y, double lambda_minus, double phi) {
    sum_lambda_minus        += lambda_minus / phi;
    sum_exp_lambda_minus_y  += exp(lambda_minus) * y / phi;
  }

};

#endif
