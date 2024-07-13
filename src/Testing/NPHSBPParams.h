#ifndef NPHSBP_PARAMS_H
#define NPHSBP_PARAMS_H

#include<RcppArmadillo.h>
#include "Node.h"
#include "functions.h"

struct NPHSBPParams {

NPHSBPParams(double scale_lambda_,
              double sigma_scale_lambda_,
              double shape_gamma_,
              double rate_gamma_,
              int K
             )
  : scale_lambda(scale_lambda_),
    sigma_scale_lambda(sigma_scale_lambda_),
    shape_gamma(shape_gamma_),
    rate_gamma(rate_gamma_)
  {
    scale_lambda_to_ab();
    gamma = arma::zeros<arma::vec>(K - 1);
  }

  arma::vec gamma;
  double sigma_scale_lambda;
  double shape_gamma;
  double rate_gamma;

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
