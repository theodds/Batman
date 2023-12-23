#ifndef QPOWER_PARAMS_H
#define QPOWER_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct QPowerParams {

  double scale_lambda_0;
  double scale_lambda;
  double sigma_scale_lambda;
  double phi;
  double p;

QPowerParams(double scale_lambda_0_,
             double scale_lambda_,
             double phi_,
             double p_) :
  scale_lambda(scale_lambda_), 
    phi(phi_),
    sigma_scale_lambda(scale_lambda_),
    scale_lambda_0(scale_lambda_0_),
    p(p_){
  }

  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_scale_lambda() const {return scale_lambda;}
  double get_phi() const {return phi;}
  double get_p() const {return p;}
  void set_scale_lambda(double x) {
    scale_lambda = x;
    scale_lambda_to_ab();
  }

};



#endif
