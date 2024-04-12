#include "CoxPEData.h"
#include "functions.h"

using namespace arma;
using namespace Rcpp;

void CoxPEData::UpdateBase() {
  
  int K = base_haz.n_elem;
  int N = Y.n_elem;

  // Update r
  r = exp(lambda_hat);
  
  // Compute the R and Fks and N
  arma::vec R  = arma::zeros<arma::vec>(K);
  arma::vec Fk = arma::zeros<arma::vec>(K);
  arma::vec Gk = arma::zeros<arma::vec>(K);
  arma::vec num_fail = arma::zeros<arma::vec>(K);

  for(int k = 0; k < K; k++) {
    num_fail(k) = 0.;
    for(int i = 0; i < bin_to_obs[k].size(); i++) {
      int idx = bin_to_obs[k][i];
      Fk(k) = Fk(k) + r(idx);
      Gk(k) = Gk(k) + (Y(idx) - time_grid(k)) * r(idx);
      num_fail(k) = num_fail(k) + delta(idx);
    }
  }

  R(0) = 0;
  for(int i = 0; i < N; i++) {
    R(0) += r(i);
  }
  R(0) = R(0) - Fk(0);

  for(int k = 1; k < K; k++) {
    R(k) = R(k-1) - Fk(k);
  }

  // Do the update for baseline hazard:
  // alpha(k) = num_fail(k) + alpha,
  // beta(k) = bin_width(k) * R(k) + G(k)
  //
  // Prior for this model sets alpha = 0 and beta = 0

  for(int k = 0; k < K; k++) {
    double alpha_k = num_fail(k);
    double beta_k  = bin_width(k) * R(k) + Gk(k);
    base_haz(k) = R::rgamma(alpha_k + shape_haz, 1.0  / (beta_k + rate_haz));
  }

  // Do the update for the cumulative hazard
  cum_base_haz(0) = base_haz(0) * bin_width(0);
  for(int k = 1; k < K; k++) {
    cum_base_haz(k) = cum_base_haz(k-1) + base_haz(k) * bin_width(k);
  }

  // Compute the baseline hazard for each Y
  for(int i = 0; i < N; i++) {
    int bin = obs_to_bin(i);
    cum_base_haz_Y(i) = (Y(i) - time_grid(bin)) * base_haz(bin);
    if(bin > 0) {
      cum_base_haz_Y(i) = cum_base_haz_Y(i) + cum_base_haz(bin - 1);
    }
  }

  // Compute the log likelihood
  loglik = 0;
  for(int i = 0; i < N; i++) {
    int bin = obs_to_bin(i);
    loglik += delta_0(i) * log(base_haz(bin) * r(i) + pop_haz(i))
      - r(i) * cum_base_haz_Y(i);
  }
  
  // arma::vec log_haz = log(base_haz);
  // for(int i = 0; i < N; i++) {
  //   int bin = obs_to_bin(i);
  //   loglik += delta(i) * log_haz(bin)
  //     + delta(i) * lambda_hat(i) 
  //     - r(i) * cum_base_haz_Y(i);
  // }
  
  // Update Shape Hazard
  // double log_shape_old = log(shape_haz);
  // double U = 0.5 * unif_rand() - 0.5;
  // double log_shape_new = log_shape_old + U;
  // double shape_new = exp(log_shape_new);
  // double sum_log_lambda = sum(log(base_haz));
  // double sum_lambda = sum(base_haz);
  // double loglik_old = (shape_haz - 1) * sum_log_lambda
  //  - base_haz.n_elem * R::lgammafn(shape_haz)
  //   + R::lgammafn(base_haz.n_elem * shape_haz)
  //   - base_haz.n_elem * shape_haz * log(sum_lambda);
  // double loglik_new = (shape_new - 1) * sum_log_lambda
  //   - base_haz.n_elem * R::lgammafn(shape_new)
  //   + R::lgammafn(base_haz.n_elem * shape_new)
  //   - base_haz.n_elem * shape_new * log(sum_lambda);
  //   shape_haz = log(unif_rand()) < loglik_new - loglik_old ? shape_new : shape_haz;
  double sum_lambda = sum(base_haz);
  shape_haz = 1;
    
  // Update Rate Hazard
  rate_haz = R::rgamma(base_haz.n_elem * shape_haz, 1.0 / (sum_lambda));
  
  // Rcout << "R: " << R << std::endl;
  // Rcout << "G: " << Gk << std::endl;
  // Rcout << "F: " << Fk << std::endl;
  // Rcout << "base_haz: " << base_haz << std::endl;
  // Rcout << "cum_base_haz: " << cum_base_haz << std::endl;
  // Rcout << "r: " << r << std::endl;
  // Rcout << "lambda_hat: " << lambda_hat << std::endl;
}

// This function recomputes various things in Data to account for updating
// the status indicator delta
void CoxPEData::RelSurvDA() {
  
  int N = delta.n_elem;
  
  for(int i = 0; i < N; i++) {
    if(delta_0(i) != 0) {
      int bin = obs_to_bin(i);
      double excess_hazard = base_haz(bin) * r(i);
      double population_hazard = pop_haz(i); 
      double p = excess_hazard / (excess_hazard + population_hazard);
      delta(i) = R::unif_rand() < p ? 1 : 0;
    }
  }
}
