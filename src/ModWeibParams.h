#ifndef MOD_WEIB_PARAMS_H
#define MOD_WEIB_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct ModWeibParams {

    ModWeibParams(double sigma_mu_, 
                  double scale_sigma_mu_, 
                  double length_scale_, 
                  double shape_length_scale_, 
                  double rate_length_scale,
                  double alpha_probit_)
    : sigma_mu(sigma_mu_), 
      scale_sigma_mu(scale_sigma_mu_), 
      length_scale(length_scale_), 
      shape_length_scale(shape_length_scale_),
      rate_length_scale(rate_length_scale_),
      alpha_probit(alpha_probit_)
    {
        prec_mu = pow(sigma_mu, -2.0);
        prec_y = 1;
        sigma_y = 1;
    }

    double get_sigma_y() const {return sigma_y;}
    double get_prec_y() const {return prec_y;}
    double get_sigma_mu() const {return sigma_mu;}
    double get_prec_mu() const {return prec_mu;}
    void set_sigma_mu(double x){sigma_mu = x; prec_mu = 1.0/pow(sigma_mu,2);}
    void set_prec_mu(double x){prec_mu = x; sigma_mu = 1.0/pow(prec_mu,0.5);}

    // Probit parameter
    double alpha_probit;

    // Basis function hypers
    double length_scale;
    double shape_length_scale;
    double rate_length_scale;

private:
    double sigma_mu, scale_sigma_mu, prec_mu, prec_y;

};

#endif