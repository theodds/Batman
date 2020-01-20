#ifndef REGSS_H
#define REGSS_H

#include <RcppArmadillo.h>

struct RegSuffStats {
  double sum_Y_0;
  double sum_Y_1;
  double sum_Y_2;

  RegSuffStats() {
    sum_Y_0 = 0.0;
    sum_Y_1 = 0.0;
    sum_Y_2 = 0.0;
  }

  void Reset() {
    sum_Y_0 = 0.0;
    sum_Y_1 = 0.0;
    sum_Y_2 = 0.0;
  }

  void Increment(double y) {
    sum_Y_0 += 1.0;
    sum_Y_1 += y;
    sum_Y_2 += y * y;
  }

};

#endif
