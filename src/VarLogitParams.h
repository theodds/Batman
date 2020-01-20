#ifndef VAR_LOGIT_PARAMS_H
#define VAR_LOGIT_PARAMS_H

#include "MLogitParams.h"
#include "VarParams.h"

struct VarLogitParams {
  
  VarParams var_params;
  MLogitParams mlogit_params;

  VarLogitParams(double kappa,
                 double scale_log_tau,
                 double sigma_scale_log_tau,
                 double shape_tau_0,
                 double rate_tau_0,
                 double scale_kappa,
                 double scale_lambda,
                 double shape_lambda_0,
                 double rate_lambda_0,
                 int num_cat) :
  var_params(kappa, scale_log_tau, sigma_scale_log_tau, shape_tau_0, rate_tau_0,
             scale_kappa),
    mlogit_params(scale_lambda, shape_lambda_0, rate_lambda_0, num_cat)
  {;}


};

#endif
