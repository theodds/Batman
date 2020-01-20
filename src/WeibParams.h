#ifndef WEIB_PARAMS_H
#define WEIB_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct WeibParams {
  
WeibParams(double scale_lambda_,
           double shape_lambda_0_,
           double rate_lambda_0_,
           double weibull_power_
           ) :
  scale_lambda(scale_lambda_),
    sigma_scale_lambda(scale_lambda_),
    shape_lambda_0(shape_lambda_0_),
    rate_lambda_0(rate_lambda_0),
    weibull_power(weibull_power_)
  {
    lambda_0 = 0.0;
    scale_lambda_to_ab();
  }

  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_scale_lambda() const {return scale_lambda;}
  void set_scale_lambda(double x) {
    scale_lambda = x;
    scale_lambda_to_ab();
  }

  double shape_lambda_0;
  double rate_lambda_0;
  double sigma_scale_lambda;
  double lambda_0;
  double weibull_power;

private:

  double scale_lambda;
  double alpha;
  double beta;

  void scale_lambda_to_ab() {
    scale_lambda_to_alpha_beta(alpha, beta, scale_lambda);
  }

};


#endif
