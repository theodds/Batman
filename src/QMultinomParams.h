#ifndef QMNOM_PARAMS_H
#define QMNOM_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct QMultinomParams {

  double scale_lambda_0;
  double sigma_scale_lambda;
  double phi;

  QMultinomParams(double scale_lambda_0_, double scale_lambda_, double phi_) :
    scale_lambda(scale_lambda_),
    phi(phi_),
    sigma_scale_lambda(scale_lambda_),
    scale_lambda_0(scale_lambda_0_) {
    scale_lambda_to_ab();
  }

  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_scale_lambda() const {return scale_lambda;}
  double get_phi() const {return phi;}
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
