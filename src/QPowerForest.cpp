#include "QPowerForest.h"

using namespace arma;
using namespace Rcpp;

void hessian_power(arma::vec& gradient,
                   arma::mat& hessian,
                   double p,
                   double phi,
                   const arma::vec& omega,
                   const QPowerData& data) {
  int N    = data.Y.n_elem;
  vec mu   = exp(data.lambda_hat);

  double D_phi = 0.;
  double D_p = 0;
  double D_phi_p = 0;
  double g_phi = 0.;
  double g_p = 0.;
  
  gradient = zeros<vec>(2);
  hessian  = zeros<mat>(2,2);

  for(int i = 0; i < N; i++) {
    double lambda = data.lambda_hat(i);
    double R = pow(data.Y(i) - mu(i), 2) / pow(mu(i), p);
    g_phi += omega(i) * (0.5 * R / phi / phi - 0.5 / phi);
    g_p += omega(i) * (- 0.5 * lambda + 0.5 * lambda * R / phi);
    // D_phi += omega(i) * (0.5 / phi / phi - R / pow(phi, 3));
    // D_p += omega(i) * (-0.5 * lambda * lambda * R / phi);
    // D_phi_p += omega(i) * (-0.5 * lambda * R / phi / phi);
    D_phi += -0.5 * omega(i) / phi / phi;
    D_p += -0.5 * omega(i) * lambda * lambda;
    D_phi_p += -0.5 * omega(i) * lambda / phi;
  }

  gradient(0) = g_phi;
  gradient(1) = g_p;
  hessian(0,0) = D_phi;
  hessian(1,0) = D_phi_p;
  hessian(0,1) = D_phi_p;
  hessian(1,1) = D_p;
}

arma::vec PredictPois(std::vector<QPowerNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

void UpdateHypers(QPowerParams& hypers, std::vector<QPowerNode*>& trees,
                  const QPowerData& data)
{
  // UpdateSigmaMu(hypers, trees);
  int NUM_NEWTON = 10;

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

  // Get mu
  vec mu = exp(data.lambda_hat);
  
  // pseudo-likelihood update for phi
  double p = hypers.p;
  vec Z = zeros<vec>(N);

  for(int k = 0; k < NUM_NEWTON; k++) {
    for(int i = 0; i < N; i++) {
      Z(i) = pow(data.Y(i) - mu(i), 2) / pow(mu(i), p);
    }
    double phi_hat = sum(Z) / N;
    double shape_phi = 0.5 * N;
    double rate_phi = 0.5 * N * phi_hat;
    double phi = 1.0 / R::rgamma(shape_phi, 1.0 / rate_phi);

    // pseudo-likelihood update for p
    double U = 2. * unif_rand() - 1.;
    double p_prop = p * exp(0.1 * U);

    double loglik_old = R::dnorm(log(p), 0., 1., 1);
    double loglik_new = R::dnorm(log(p_prop), 0., 1., 1);
    for(int i = 0; i < N; i++) {
      loglik_old += R::dnorm(data.Y(i), mu(i), sqrt(phi * pow(mu(i), p)), 1);
      loglik_new += R::dnorm(data.Y(i), mu(i), sqrt(phi * pow(mu(i), p_prop)), 1);
    }
    U = unif_rand();
    p = log(U) < loglik_new - loglik_old ? p_prop : p;

    hypers.phi = phi;
    hypers.p = p;
  }
  
  // // Initialize variables for Newton-Rhapson
  // vec phi_p = zeros<vec>(2);
  // phi_p(0) = hypers.phi;
  // phi_p(1) = hypers.p;
  
  // for(int i = 0; i < N; i++) {
  //   Z(i) = pow(data.Y(i) - mu(i), 2) / pow(mu(i), p);
  // }
  // phi_p(0) = sum(omega % Z);

  // vec gradient;
  // mat hessian;
  
  // for(int i = 0; i < NUM_NEWTON; i++) {
  //   hessian_power(gradient, hessian, phi_p(1), phi_p(0), omega, data);
  //   phi_p = phi_p - inv_sympd(hessian) * gradient;
  //   if(phi_p(0) < 0) phi_p(0) = 0.00001;
  //   if(phi_p(1) < 0) phi_p(1) = 0.00001;
  // }

  // hypers.phi = phi_p(0);
  // hypers.p = phi_p(1);
}

// [[Rcpp::export]]
List QPowerBart(const arma::mat& X,
                const arma::vec& Y,
                const arma::mat& X_test,
                const arma::sp_mat& probs,
                int num_trees,
                double scale_lambda,
                double scale_lambda_0,
                int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  QPowerParams pois_params(scale_lambda_0, scale_lambda, 1.0, .9999);
  QPowerForest forest(num_trees, &tree_hypers, &pois_params);
  QPowerData data(X,Y);
  mat lambda = zeros<mat>(num_save, Y.size());
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec phi = zeros<vec>(num_save);
  vec p = zeros<vec>(num_save);

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
    p(iter) = pois_params.get_p();
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;
  out["phi"] = phi;
  out["p"] = p;

  return out;
}
