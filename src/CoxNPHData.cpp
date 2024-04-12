#include "CoxNPHData.h"
#include "functions.h"

using namespace arma;
using namespace Rcpp;

void CoxNPHData::UpdateBase() {
  int DO_COL = 0;
  
  int K = base_haz.n_elem;
  int N = Y.n_elem;

  // Update r
  r = exp(lambda_hat);
  
  // Shape and Rate Parameters
  arma::vec A = arma::zeros<arma::vec>(K);
  // arma::vec B = trans(sum(Z % r, DO_COL));
  arma::vec B = arma::zeros<arma::vec>(K);
  for(int i = 0; i < N; i++) {
    for(int k = 0; k < K; k++) {
      B(k) += Z(i,k) * r(i,k);
    }
  }
  for(int i = 0; i < N; i++) {
    if(delta(i) != 0) {
      int b = obs_to_bin(i);
      A(b) += 1.;
    }
  }

  // Update the lambda
  for(int b = 0; b < K; b++) {
    double shape_up = A(b) + shape_haz;
    double rate_up = B(b) + rate_haz;
    base_haz(b) = R::rgamma(shape_up, 1. / rate_up);
    // base_haz(b) = 1.;
  }
  
  // Do the update for the cumulative hazard
  //   cum_base_haz(0) = base_haz(0) * bin_width(0);
  //   for(int k = 1; k < K; k++) {
  //     cum_base_haz(k) = cum_base_haz(k-1) + base_haz(k) * bin_width(k);
  //   }

  //   // Compute the baseline hazard for each Y
  //   for(int i = 0; i < N; i++) {
  //     int bin = obs_to_bin(i);
  //     cum_base_haz_Y(i) = (Y(i) - time_grid(bin)) * base_haz(bin);
  //     if(bin > 0) {
  //       cum_base_haz_Y(i) = cum_base_haz_Y(i) + cum_base_haz(bin - 1);
  //     }
  //   }
  
  // Compute the log likelihood
    loglik = 0;
    arma::vec haz_eval = (Z % r) * base_haz;
    for(int i = 0; i < N; i++) {
      int bin = obs_to_bin(i);
      loglik += delta_0(i) * log(base_haz(bin) * r(i,bin) + pop_haz(i))
        - haz_eval(i);
    }
  
  // Update Rate Hazard
    double sum_lambda = sum(base_haz);
    shape_haz = 1.;
    rate_haz = R::rgamma(base_haz.n_elem * shape_haz + 1, 1.0 / (sum_lambda + 1));
}
  
void CoxNPHData::RelSurvDA() {
  int N = delta.n_elem;
  for(int i = 0; i < N; i++) {
    if(delta_0(i) != 0) {
      int bin = obs_to_bin(i);
      double excess_hazard = base_haz(bin) * r(i, bin);
      double population_hazard = pop_haz(i); 
      double p = excess_hazard / (excess_hazard + population_hazard);
      delta(i) = R::unif_rand() < p ? 1 : 0;
    }
  }
}
