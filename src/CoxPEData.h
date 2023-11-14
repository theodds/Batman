#ifndef COX_PE_DATA_H
#define COX_PE_DATA_H

#include <RcppArmadillo.h>

struct CoxPEData {

  // Raw data
  arma::mat X;
  arma::vec Y;
  arma::uvec delta;
  arma::uvec delta_0; // This is the data status indicator; is fixed across all iterations
  std::vector<std::vector<int>> bin_to_obs;
  arma::uvec obs_to_bin;
  arma::vec time_grid;
  arma::vec bin_width;
  arma::vec pop_haz;

  // Derived quantities
  arma::vec r; // The exp(lambda_hat)'s
  arma::vec lambda_hat; // The lambdas, given by output of function
  arma::vec base_haz;
  arma::vec cum_base_haz;
  arma::vec cum_base_haz_Y;
  double loglik;

  CoxPEData(const arma::mat& X_,
            const arma::vec& Y_,
            const arma::uvec delta_,
            std::vector<std::vector<int>> bin_to_obs_,
            const arma::uvec& obs_to_bin_,
            const arma::vec& time_grid_,
            const arma::vec& bin_width_,
            const arma::vec& base_haz_init,
            const arma::vec& pop_haz_
          ) {

    X = X_;
    Y = Y_;
    delta = delta_;
    delta_0 = delta_;
    bin_to_obs = bin_to_obs_;
    obs_to_bin = obs_to_bin_;
    time_grid = time_grid_;
    bin_width = bin_width_;
    base_haz   = base_haz_init;
    pop_haz   = pop_haz_;
    cum_base_haz= arma::zeros<arma::vec>(base_haz.n_elem);
    cum_base_haz_Y = arma::zeros<arma::vec>(Y.n_elem);
    
    int K = base_haz.n_elem;
    int N = Y.n_elem;

    lambda_hat = arma::zeros<arma::vec>(Y.size());
    r          = exp(lambda_hat);
    

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

    loglik = 0;

  }
  
  void UpdateBase();
  void RelSurvDA();
  
};

#endif
