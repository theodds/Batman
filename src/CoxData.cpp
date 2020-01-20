#include "CoxData.h"
#include "functions.h"

using namespace arma;
using namespace Rcpp;

void UpdatePhi(CoxData& data) {
  
  double r_0 = 0.0;
  for(int i = (data.Y.size() - 1); i >= 0; i--) {
    int j = data.order(i);
    r_0 += exp(data.lambda_hat(j));
    data.phi(j) = R::rgamma(1, 1.0/r_0) * data.delta(j);
  }
  data.ComputeR();
}

void CoxData::Shuffle() 
{

  vec eta = zeros<vec>(X.n_rows + 1);
  eta(Y.size()) = R_NegInf;
  
  for(int k = L.size() - 1; k >= 0; k--) {
    int u = U(k);
    int l = L(k);
    double loglik_after = 0.;
    double loglik_before = 0.;
    for(int i = u; i >= l; i--) {
      int j = order(i);
      loglik_before += lambda_hat(j);
      eta(i) = log_sum_exp(eta(i+1), lambda_hat(j));
      loglik_before -= eta(i);
    }

    int i1 = sample_class(u - l + 1) + l;
    int i2 = sample_class(u - l + 1) + l;
    int o1 = order(i1); int o2 = order(i2); order(i1) = o2; order(i2) = o1;
    
    for(int i = u; i >= l; i--) {
      int j = order(i);
      loglik_after += lambda_hat(j);
      eta(i) = log_sum_exp(eta(i+1), lambda_hat(j));
      loglik_after -= eta(i);
    }
    if(log(unif_rand()) > loglik_after - loglik_before) {
      int o1 = order(i1); int o2 = order(i2); order(i1) = o2; order(i2) = o1;
    }
  }
  UpdatePhi(*this);
}