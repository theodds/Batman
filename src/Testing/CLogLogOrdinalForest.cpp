#include "GammaRegForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictPois(std::vector<GammaNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

void UpdateHypers(GammaParams& hypers, std::vector<GammaNode*>& trees,
                  const GammaData& data)
{
  int N = data.Y.n_elem;

  // Create vector of means and Z's and compute phi_hat
  vec mu = zeros<vec>(N);
  vec Z = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    mu(i) = exp(-data.lambda_hat(i));
    Z(i) = pow(data.Y(i) - mu(i), 2) / pow(mu(i), 2);
  }
  
  double phi_hat = mean(Z);
  
  // Get alpha and an approximation of its standard deviation
  double alpha = 1.0 / hypers.phi;
  double sigma_alpha = 1.0 / phi_hat * sqrt(2.0 / N);
  
  // Get sufficient statistics
  double S = sum(log(data.Y / mu));
  double R = sum(data.Y / mu);

  // Get original likelihood
  double loglik_old = N * alpha * log(alpha) + alpha * (S - R) - 
    N * R::lgammafn(alpha);
  
  for(int k = 0; k < 10; k++) {
    double alpha_prop = alpha + (2. * unif_rand() - 1) * sigma_alpha;
    if(alpha_prop < 0) continue;
    
    double loglik_new = N * alpha_prop * log(alpha_prop) 
      + alpha_prop * (S - R) - N * R::lgammafn(alpha_prop);
    
    alpha = unif_rand() < loglik_new - loglik_old ? alpha_prop : alpha;
  }

  hypers.phi = 1.0 / alpha;
}

// [[Rcpp::export]]
List GammaRegBart(const arma::mat& X,
                  const arma::vec& Y,
                  const arma::mat& X_test,
                  const arma::sp_mat& probs,
                  int num_trees,
                  double scale_lambda,
                  double scale_lambda_0,
                  int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  GammaParams pois_params(scale_lambda_0, scale_lambda, 1.0);
  GammaForest forest(num_trees, &tree_hypers, &pois_params);
  GammaData data(X,Y);
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
  out["alpha"] = 1 / phi;

  return out;
}
