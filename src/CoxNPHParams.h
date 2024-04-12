#ifndef COX_NPH_PARAMS_H
#define COX_NPH_PARAMS_H

#include<RcppArmadillo.h>
#include "Node.h"
#include "functions.h"

struct CoxNPHParams {

CoxNPHParams(double scale_lambda_,
          double sigma_scale_lambda_)
  : scale_lambda(scale_lambda_),
    sigma_scale_lambda(sigma_scale_lambda_)
  {
    scale_lambda_to_ab();
  }


  double sigma_scale_lambda;

  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_scale_lambda() const {return scale_lambda;}
  void set_scale_lambda(double x) {
    scale_lambda = x;
    scale_lambda_to_ab();
  }

private:

  double scale_lambda;
  double alpha;
  double beta;

  void scale_lambda_to_ab() {
    scale_lambda_to_alpha_beta(alpha, beta, scale_lambda);
  }

};

#endif
