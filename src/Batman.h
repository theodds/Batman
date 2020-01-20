#ifndef BATMAN_H
#define BATMAN_H

#include "VarLogitForest.h"

struct MixParams {

  // Actual mixture parameters
  arma::vec loc_class;
  arma::vec prec_class;

  // location hyperparameters
  double sigma_loc;

  // Precision hyperparameter
  double shape_prec; // prec ~ Gamma(shape, shape), and we set shape ~ Gam(4,1/5) by default.
  double shape_shape;
  double rate_shape;


  // NOT NEEDED: taken care of by Forest
  // Hyperparameter for log_weight, will be log Gam(shape,1) with a small shape. */
  // arma::vec log_weight_class;
  // double shape_weight;

  MixParams(int k) {
    sigma_loc = 2;
    shape_shape = 4.0;
    rate_shape = 0.2;
    shape_prec = R::rgamma(shape_shape, 1.0/rate_shape);
    loc_class = arma::zeros<arma::vec>(k);
    prec_class = arma::ones<arma::vec>(k);
    // shape_weight = 1.0 / ((double)k);
    // log_weight_class = arma::zeros<arma::vec>(k);
    for(int i = 0; i < k; i++) {
      loc_class(i) = R::rnorm(0.0, sigma_loc);
      /* prec_class(i) = R::rgamma(shape_prec, shape_prec); */
    }
  }
};

arma::vec PredictMixLoc(const arma::uvec& clust, MixParams& params);
arma::vec PredictMixPrec(const arma::uvec& clust, MixParams& params);

#endif
