#ifndef COX_NPH_SS_H
#define COX_NPH_SS_H

struct CoxNPHSuffStats {
  double sum_delta;
  double sum_Z_haz_r;

  CoxNPHSuffStats() {
    Reset();
  }

  void Reset() {
    sum_delta                         = 0.;
    sum_Z_haz_r = 0.;
  }

  void Increment(double delta_b, double lambda_minus, double Z, double base_haz)
  {
    sum_delta += delta_b;
    sum_Z_haz_r += base_haz * Z * exp(lambda_minus);
  }
};

#endif
