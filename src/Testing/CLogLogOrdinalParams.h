#ifndef CLOGLOGORDINAL_PARAMS_H
#define CLOGLOGORDINAL_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct CLogLogOrdinalParams {

  double sigma_scale_lambda;
  arma::vec gamma;
  double alpha_gamma;
  double beta_gamma;

  CLogLogOrdinalParams(double sigma_scale_lambda_,
                       double alpha_gamma_,
                       double beta_gamma_,
                       int num_levels) :
  sigma_scale_lambda(sigma_scale_lambda_),
    scale_lambda(sigma_scale_lambda_),
    alpha_gamma(alpha_gamma_),
    beta_gamma(beta_gamma_){
    gamma = arma::zeros<arma::vec>(num_levels - 1);
    scale_lambda_to_ab();
  }

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
