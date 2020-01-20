#ifndef WEIBSS_H
#define WEIBSS_H

#include <RcppArmadillo.h>

struct WeibSuffStats {

  double num_W;
  double sum_Y_elam;
  double sum_log_W;
  double sum_lam_num_W;

  void Reset() {
    num_W = 0.;
    sum_Y_elam = 0.;
    sum_log_W = 0.;
    sum_lam_num_W = 0.;
  }

  WeibSuffStats() {
    Reset();
  }


  void Increment(double y_elam,
                 double num_w,
                 double sum_log_w,
                 double lam_num_w) {
    num_W += num_w;
    sum_Y_elam += y_elam;
    sum_log_W += sum_log_w;
    sum_lam_num_W += lam_num_w;
  }
};


#endif
