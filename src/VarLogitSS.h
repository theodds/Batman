#ifndef VAR_LOGIT_SS
#define VAR_LOGIT_SS

#include <RcppArmadillo.h>
#include "VarSS.h"
#include "MLogitSS.h"

struct VarLogitSuffStats {
  
  VarSuffStats var_stats;
  MLogitSuffStats mlogit_stats;

VarLogitSuffStats(int num_cat) : var_stats(), mlogit_stats(num_cat)
  {
    ;
  }

  void Reset() {
    var_stats.Reset();
    mlogit_stats.Reset();
  }

  void IncrementVar(double y, double eta) {
    var_stats.Increment(y, eta);
  }

  void IncrementLogit(unsigned int j,
                      const arma::vec& exp_lambda_minus,
                      double phi) {
    mlogit_stats.Increment(j, exp_lambda_minus, phi);
  }

};

#endif
