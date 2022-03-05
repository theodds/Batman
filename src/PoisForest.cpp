#include "PoisForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictPois(std::vector<PoisNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

void UpdateHypers(PoisParams& hypers, std::vector<PoisNode*>& trees,
                  const PoisData& data)
{
  // UpdateSigmaY(hypers, data);
  // UpdateSigmaMu(hypers, trees);
}

// [[Rcpp::export]]
List PoisBart(const arma::mat& X,
              const arma::vec& Y,
              const arma::mat& X_test,
              const arma::sp_mat& probs,
              int num_trees,
              double scale_lambda,
              double scale_lambda_0,
              int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  PoisParams pois_params(scale_lambda_0, scale_lambda);
  PoisForest forest(num_trees, &tree_hypers, &pois_params);
  PoisData data(X,Y);
  mat lambda = zeros<mat>(num_save, Y.size());
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, pois_params, tree_hypers);
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, pois_params, tree_hypers);
    }
    lambda.row(iter) = trans(data.lambda_hat);
    lambda_test.row(iter) = trans(PredictPois(forest.trees, X_test));
    counts.row(iter) = trans(get_var_counts(forest.trees));
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;

  return out;
}
