#ifndef WEIB_DATA_H
#define WEIB_DATA_H

#include <RcppArmadillo.h>

struct WeibData {
  arma::mat X;
  arma::vec Y;
  std::vector<std::vector<double>> W;
  arma::vec mu_hat;

  WeibData(const arma::mat& X_,
           const arma::vec& Y_,
           const arma::vec& W_,
           const arma::uvec& idx_) {
    X = X_;
    Y = Y_;
    mu_hat = arma::zeros<arma::vec>(X.n_rows);
    W.clear();
    W.resize(Y.size());
    for(int i = 0; i < W.size(); i++) {
      W[i].clear();
    }
    for(int i = 0; i < W_.size(); i++) {
      int idx = idx_(i);
      double w = W_(i);
      W[idx].push_back(w);
    }
  }
};

#endif
