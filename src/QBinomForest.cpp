#include "QBinomForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictPois(std::vector<QBinomNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

void UpdateHypers(QBinomParams& hypers, std::vector<QBinomNode*>& trees,
                  QBinomData& data)
{
  int N = data.X.n_rows;

  /*Step 1: Update rho's*/

  for(int i = 0; i < N; i++) {
    double a = data.n(i) / hypers.phi;
    double b = 1.0 + exp(data.lambda_hat(i));
    data.rho(i) = R::rgamma(a, 1.0 / b);
  }
  
  /*Step 2: BBQ Update for phi*/
  // Create the Bayesian bootstrap vector
  vec omega = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    omega(i) = R::rexp(1.0);
  }
  double omega_sum = sum(omega);
  for(int i = 0; i < N; i++) {
    omega(i) = omega(i) / omega_sum;
  }

  // Create vector of means and Z's
  vec Z = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    double mu = expit(data.lambda_hat(i));
    double p = data.Y(i) / data.n(i);
    double n = data.n(i);
    Z(i) = n * pow(p - mu, 2) / mu / (1. - mu);
  }

  // Update phi
  hypers.phi = sum(Z % omega);
}

// [[Rcpp::export]]
List QBinomBart(const arma::mat& X,
                const arma::vec& Y,
                const arma::vec& n,
                const arma::mat& X_test,
                const arma::sp_mat& probs,
                int num_trees,
                double scale_lambda,
                double scale_lambda_0,
                int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  QBinomParams pois_params(scale_lambda_0, scale_lambda, 1.0);
  QBinomForest forest(num_trees, &tree_hypers, &pois_params);
  QBinomData data(X,Y);
  mat lambda = zeros<mat>(num_save, Y.size());
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec phi = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, pois_params, tree_hypers);
    if(iter % 100 == 0) Rcout << "\rFinishing warmup iteration "
                              << iter << "\t\t\t";
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, pois_params, tree_hypers);
    }
    if(iter % 100 == 0) Rcout << "\rFinishing save iteration "
                              << iter << "\t\t\t";
    lambda.row(iter) = trans(data.lambda_hat);
    lambda_test.row(iter) = trans(PredictPois(forest.trees, X_test));
    counts.row(iter) = trans(get_var_counts(forest.trees));
    phi(iter) = pois_params.get_phi();
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;
  out["phi"] = phi;

  return out;
}
