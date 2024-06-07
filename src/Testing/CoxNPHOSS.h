#ifndef COX_NPHO_SS_H
#define COX_NPHO_SS_H

struct CoxNPHOSuffStats {
  double A_l;
  double B_l;
  
  CoxNPHOSuffStats() {
    Reset();
  }

  void Reset() {
    A_l = 0.;
    B_l = 0.;
  }

  void Increment(double delta_b, double Z, double lambda_minus, double gamma)
  {
    A_l += delta_b;
    B_l += Z * exp(lambda_minus + gamma);
  }
};

#endif
