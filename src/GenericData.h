#ifndef GENERIC_DATA_H
#define GENERIC_DATA_H

#include <RcppArmadillo.h>

struct GenericData {

  // Raw data
  arma::mat X;
  arma::vec Y;
  arma::uvec delta;
  arma::uvec order;
  arma::uvec L;
  arma::uvec U;

  // Derived quantities
  arma::vec phi;
  arma::vec r;
  arma::vec lambda_hat;

  // TODO Make sure this is correct; also make sure this is called after Phi is updated!
  void ComputeR() {

    int N = order.size();
    double r_0 = 0.0;
    for(int i = 0; i < N; i++) {
      int o = order(i);
      r_0 += phi(o) * delta(o);
      r(o) = r_0;
    }
  }
  
  void Shuffle();

  CoxData(const arma::mat& X_,
          const arma::vec& Y_,
          const arma::uvec delta_,
          const arma::uvec order_,
          const arma::uvec L_, 
          const arma::uvec U_
          ) {

    X = X_;
    Y = Y_;
    delta = delta_;
    order = order_;
    L = L_;
    U = U_;

    phi = arma::ones<arma::vec>(Y.size());
    lambda_hat = arma::zeros<arma::vec>(Y.size());
    r = arma::zeros<arma::vec>(Y.size());

    ComputeR();


  }

};

void UpdatePhi(CoxData& data);

#endif
