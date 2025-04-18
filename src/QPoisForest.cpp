#include "QPoisForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictPois(std::vector<QPoisNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

arma::vec QPoisForest::do_predict(const arma::mat& X) {
  return PredictPois(trees, X);
}


void UpdateHypers(QPoisParams& hypers, std::vector<QPoisNode*>& trees,
                  const QPoisData& data)
{
  // UpdateSigmaY(hypers, data);
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
    mu(i) = exp(data.lambda_hat(i));
    Z(i) = pow(data.Y(i) - mu(i), 2) / mu(i);
  }

  // Update phi
  hypers.phi = sum(Z % omega);
  // double phi_hat = sum(Z) / N;
  // hypers.phi = R::rgamma(0.5 * N, 2. * phi_hat / N);

}

// [[Rcpp::export]]
List QPoisBart(const arma::mat& X,
              const arma::vec& Y,
              const arma::mat& X_test,
              const arma::sp_mat& probs,
              int num_trees,
              double scale_lambda,
              double scale_lambda_0,
              int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  QPoisParams pois_params(scale_lambda_0, scale_lambda, 1.0);
  QPoisForest forest(num_trees, &tree_hypers, &pois_params);
  QPoisData data(X,Y);
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

Rcpp::List  QPoisForest::do_gibbs(const arma::mat& X,
                                  const arma::vec& Y,
                                  const arma::vec& offset,
                                  const arma::mat& X_test,
                                  int num_iter)


{
  mat lambda_out = zeros<mat>(num_iter, X_test.n_rows);
  List out;
  QPoisData data(X,Y);
  data.lambda_hat = do_predict(X) + offset;
  for(int i = 0; i < num_iter; i++) {
    IterateGibbs(trees, data, *params, *tree_hypers);
    lambda_out.row(i) = trans(do_predict(X_test));
  }
  
  out["lambda"] = lambda_out;
  return out;
}



RCPP_MODULE(qpois_forest) {

  class_<QPoisForest>("QPoisForest")
    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &QPoisForest::do_gibbs)
    .method("get_s", &QPoisForest::get_s)
    .method("get_counts", &QPoisForest::get_counts)
    .method("get_sigma_mu", &QPoisForest::get_sigma_mu)
    .method("get_phi", &QPoisForest::get_phi)
    .method("set_phi", &QPoisForest::set_phi)
    .method("do_predict", &QPoisForest::do_predict)
    // .method("get_tree_counts", &Forest::get_tree_counts)
    // .method("predict_iteration", &Forest::predict_iteration)
    ;

  
}
