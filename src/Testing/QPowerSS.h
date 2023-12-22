#ifndef QPOWER_SS_H
#define QPOWER_SS_H

#include <RcppArmadillo.h>

struct QPowerSuffStats {
  double A;
  double B;

  QPowerSuffStats() {
    Reset();
  }

  void Reset() {
    A = 0.;
    B = 0.;
  }

  void Increment(double y, double lambda_minus, double phi, double p) {
    A += y * exp(lambda_minus * (1. - p));
    B += exp(lambda_minus * (2. - p));
  }
};

#endif
