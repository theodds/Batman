#ifndef CLOGLOGORDINAL_SS_H
#define CLOGLOGORDINAL_SS_H

#include <RcppArmadillo.h>

struct CLogLogOrdinalSuffStats {
  double sum_Y_less_K;
  double other_sum;

  CLogLogOrdinalSuffStats() {
    Reset();
  }

  void Reset() {
    sum_Y_less_K = 0.;
    other_sum = 0.;
  }

  void Increment(unsigned int y,
                 double lambda_minus,
                 double Z,
                 const arma::vec& gamma,
                 const arma::vec& seg) {
    if(y < gamma.size()) {
      sum_Y_less_K += 1.;
      other_sum += exp(lambda_minus) * (Z * exp(gamma(y)) + seg(y));
    } else {
      other_sum += exp(lambda_minus) * seg(y); 
    }
  }
};

#endif
