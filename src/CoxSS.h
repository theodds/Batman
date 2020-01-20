#ifndef COX_SS_H
#define COX_SS_H

struct CoxSuffStats {
  double sum_delta;
  double sum_delta_lambda_minus;
  double sum_r_exp_lambda_minus;

  CoxSuffStats() {
    Reset();
  }

  void Reset() {
    sum_delta              = 0.;
    sum_delta_lambda_minus = 0.;
    sum_r_exp_lambda_minus = 0.;
  }

  void Increment(double delta, double lambda_minus, double r)
  {
    sum_delta              += delta;
    sum_delta_lambda_minus += delta * lambda_minus;
    sum_r_exp_lambda_minus += r * exp(lambda_minus);
  }

};

#endif
