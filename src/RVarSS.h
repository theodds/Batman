#ifndef RVAR_SS_
#define RVAR_SS_H

#include <RcppArmadillo.h>

struct RVarSuffStats {

  double sum_eta;
  double sum_log_eta;
  double sum_eta_y;
  double sum_eta_y_sq;
  int n; 

  RVarSuffStats() {
    Reset();
  }

  void Reset() {
    sum_eta = 0.;
    sum_log_eta = 0.;
    sum_eta_y = 0.;
    sum_eta_y_sq = 0.;
    n = 0;
  }

  void Increment(double y, double eta) {
    sum_eta += eta;
    sum_eta_y += y * eta;
    sum_eta_y_sq += eta * y * y;
    sum_log_eta += log(eta);
    n += 1;
  }

};


#endif
