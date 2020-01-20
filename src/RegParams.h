#ifndef REG_PARAMS_H
#define REG_PARAMS_H

#include <RcppArmadillo.h>
#include "Node.h"

struct RegParams {

  RegParams(double sigma_y_,
            double sigma_mu_,
            double scale_sigma_,
            double scale_sigma_mu_)
  : sigma_y(sigma_y_),
    sigma_mu(sigma_mu_),
    scale_sigma(scale_sigma_),
    scale_sigma_mu(scale_sigma_mu_)
  {
    prec_y = pow(sigma_y, -2.0);
    prec_mu = pow(sigma_mu, -2.0);
  }

  double scale_sigma;
  double scale_sigma_mu;

  double get_sigma_y() const {return sigma_y;}
  double get_prec_y() const {return prec_y;}
  double get_sigma_mu() const {return sigma_mu;}
  double get_prec_mu() const {return prec_mu;}
  void set_sigma_y(double x){sigma_y = x; prec_y = 1.0/pow(sigma_y, 2);}
  void set_sigma_mu(double x){sigma_mu = x; prec_mu = 1.0/pow(sigma_mu,2);}
  void set_prec_y(double x){prec_y = x; sigma_y = 1.0/pow(prec_y,0.5);}
  void set_prec_mu(double x){prec_mu = x; sigma_mu = 1.0/pow(prec_mu,0.5);}

private:
  double sigma_y;
  double prec_y;
  double sigma_mu;
  double prec_mu;
};


#endif
