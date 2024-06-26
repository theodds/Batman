#ifndef VAR_PARAMS_H
#define VAR_PARAMS_H

#include<RcppArmadillo.h>
#include "Node.h"
#include "functions.h"

struct VarParams {

VarParams(double kappa_,
          double scale_log_tau_,
          double sigma_scale_log_tau_,
          double shape_tau_0_,
          double rate_tau_0_,
          double scale_kappa_) :
  kappa(kappa_), scale_log_tau(scale_log_tau_),
    sigma_scale_log_tau(sigma_scale_log_tau_),
    shape_tau_0(shape_tau_0_), rate_tau_0(rate_tau_0_),
    scale_kappa(scale_kappa_)
  {
    tau_0 = 1.0;
    scale_lambda_to_ab();
  }

  double kappa;
  double shape_tau_0;
  double rate_tau_0;
  double tau_0;
  double scale_kappa;
  double sigma_scale_log_tau;

  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_scale_log_tau() const {return scale_log_tau;}
  void set_scale_log_tau(double x) {scale_log_tau = x; scale_lambda_to_ab();}

private:

  double scale_log_tau;
  double alpha;
  double beta;

  void scale_lambda_to_ab() {
    scale_lambda_to_alpha_beta(alpha, beta, scale_log_tau);
  }

};

#endif
