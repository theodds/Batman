#ifndef NPHSBP_DATA_H
#define NPHSBP_DATA_H

#include <RcppArmadillo.h>

struct NPHSBPData {

  // Raw data
  arma::mat X;
  arma::vec Y;
  arma::sp_mat Z;
  std::vector<std::vector<int>> bin_to_obs;
  
  // Derived quantities
  arma::mat r; // The exp(lambda_hat)'s
  arma::mat lambda_hat; // The lambdas, given by output of function
  double loglik;
  
  NPHSBPData(const arma::mat& X_,
             const arma::vec& Y_,
             std::vector<std::vector<int>> bin_to_obs_
             ) {



    X = X_;
    Y = Y_;
    bin_to_obs = bin_to_obs_;
    int K = bin_to_obs.size();
    int N = Y.n_elem;
    // Set Z
    Z = arma::zeros<arma::sp_mat>(N, K - 1);
    for(int i = 0; i < N; i++) {
      // The first K - 1 bins
      for(int k = 0; k < (K-1); k++) {
        if(Y(i) == k) {
          Z(i,k) = 0.5;
        }
        if(Y(i) > k) {
          Z(i,k) = 1;
        }
      }
    }

    lambda_hat = arma::zeros<arma::mat>(N, K - 1);
    r          = exp(lambda_hat);
    loglik = 0;
  }
};

#endif
