#ifndef COX_PE_SS_H
#define COX_PE_SS_H

struct CoxPESuffStats {
  double sum_delta;
  double sum_exp_lambda_times_cum_base_haz;

  CoxPESuffStats() {
    Reset();
  }

  void Reset() {
    sum_delta                         = 0.;
    sum_exp_lambda_times_cum_base_haz = 0.;
  }

  void Increment(double delta, double lambda_minus, double cum_base_haz)
  {
    sum_delta += delta;
    sum_exp_lambda_times_cum_base_haz += cum_base_haz * exp(lambda_minus);
  }
};

#endif
