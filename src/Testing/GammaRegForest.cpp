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
  // UpdateSigmaMu(hypers, trees);

  // Create the Bayesian bootstrap vector
  int N = data.X.n_rows;
  vec omega = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    omega(i) = R::rexp(1.0);
  }
  double omega_sum = sum(omega);
  for(int i = 0; i < N; i++) {
    omega(i) = omega(i) / omega_sum;
  }

  // Create vector of means and Z's
  vec mu = zeros<vec>(N);
  vec Z = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    mu(i) = exp(-data.lambda_hat(i));
    Z(i) = pow(data.Y(i) - mu(i), 2) / pow(mu(i), 2);
  }

  // Update phi
  hypers.phi = sum(Z % omega);
  // Rcout << "hypers.phi = sum(Z % omega);" << "    " << hypers.phi << std::endl;
  // double phi_hat = sum(Z) / N;
  // hypers.phi = R::rgamma(0.5 * N, 2. * phi_hat / N);

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
