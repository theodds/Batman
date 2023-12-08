#include "QMultinomForest.h"

using namespace arma;
using namespace Rcpp;

arma::mat Predict(std::vector<QMultinomNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  int K = forest[0]->lambda.n_elem;
  mat out = zeros<mat>(X.n_rows, K);
  for(int n = 0; n < N; n++) {
    out = out + Predict(forest[n], X);
  }
  return out;
}

double fast_YtVinvY(arma::rowvec& y, arma::rowvec& mu) {
  int K = y.n_elem;
  double term_1 = 0;
  double term_2 = 0;
  for(int k = 0; k < K - 1; k++) {
    term_1 += pow(y(k) - mu(k), 2) / mu(k);
    term_2 += y(k) - mu(k);
  }
  term_2 = pow(term_2, 2) / mu(K - 1);
  return term_1 + term_2;
}

void UpdateHypers(QMultinomParams& hypers, std::vector<QMultinomNode*>& trees,
                  QMultinomData& data)
{
  int N = data.X.n_rows;
  int K = data.Y.n_cols;

  // Compute the mean
  arma::mat mu = exp(data.lambda_hat);
  arma::vec sum_exp = arma::vec<arma::zeros>(N);
  for(int i = 0; i < N; i++) {
    sum_exp(i) = sum(mu.row(i));
    mu.row(i) = mu.row(i) / sum_exp(i);
  }

  /*Step 1: Update rho's*/
  for(int i = 0; i < N; i++) {
    double a = data.n(i) / hypers.phi;
    double b = sum_exp(i);
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

  // Compute phi_hat
  vec phi_hat = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    phi_hat(i) = data.n(i) * fast_YtVinvY(data.Y.row(i), mu.row(i)) / (K - 1);
  }
  
  // Update phi
  hypers.phi = sum(phi_hat % omega);
}

// [[Rcpp::export]]
List QMultinomBart(const arma::mat& X,
                const arma::mat& Y,
                const arma::vec& n,
                const arma::mat& X_test,
                const arma::sp_mat& probs,
                int num_trees,
                double scale_lambda,
                double scale_lambda_0,
                int num_burn, int num_thin, int num_save)
{

  int N = Y.n_rows; 
  int K = Y.n_cols;
  int N_test = X_test.n_rows;
  // Rcout << "TreeHypers tree_hypers(probs);" << std::endl;
  TreeHypers tree_hypers(probs);
  QMultinomParams mnom_params(scale_lambda_0, scale_lambda, 1.0);
  QMultinomForest forest(num_trees, &tree_hypers, &pois_params);
  QMultinomData data(X,Y, n);
  cube lambda = zeros<cube>(N, K, num_save);
  cube lambda_test = zeros<mat>(N_test, K, num_save);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec phi = zeros<vec>(num_save);

  // Rcout << "for(int iter = 0; iter < num_burn; iter++);" << std::endl;
  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, mnom_params, tree_hypers);
    if(iter % 100 == 0)
      Rcout << "\rFinishing warmup iteration " << iter << "\t\t\t";
    // Rcout << data.rho << std::endl;
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, pois_params, tree_hypers);
    }
    if(iter % 100 == 0) Rcout << "\rFinishing save iteration "
                              << iter << "\t\t\t";
    lambda.slice(iter) = data.lambda_hat;
    lambda_test.slice(iter) = Predict(forest.trees, X_test);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    phi(iter) = mnom_params.get_phi();
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;
  out["phi"] = phi;

  return out;
}
