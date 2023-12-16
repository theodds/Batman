#ifndef RVAR_PARAMS_H
#define RVAR_PARAMS_H

#include<RcppArmadillo.h>
#include "Node.h"
#include "functions.h"

struct RVarParams {

RVarParams(double scale_log_tau_,
           double sigma_scale_log_tau_,
           double shape_tau_0_,
           double rate_tau_0_,
           bool update_scale_log_tau_) :
  scale_log_tau(scale_log_tau_),
    sigma_scale_log_tau(sigma_scale_log_tau_),
    shape_tau_0(shape_tau_0_), rate_tau_0(rate_tau_0_),
    update_scale_log_tau(update_scale_log_tau_)
  {
    tau_0 = 1.0;
    scale_lambda_to_ab();
  }

  double shape_tau_0;
  double rate_tau_0;
  double tau_0;
  double sigma_scale_log_tau;
  bool update_scale_log_tau;

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
