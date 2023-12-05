#include "QNBForest.h"

using namespace arma;
using namespace Rcpp;

void V_eta(double mu, double eta, double& V, double& Vp, double& Vpp) {
  // double l = log(mu)
  // V = pow(mu, eta);
  // Vp = l * V;
  // Vpp = l * Vp;
  V = mu + eta * mu * mu;
  Vp = mu * mu;
  Vpp = 0.;
}

double R_eta(double eta, double phi, const arma::vec& mu_vec,
             const arma::vec& omega, const QNBData& data,
             double& R, double& Rp, double& Rpp) {
  double V, Vp, Vpp;
  int N = omega.n_elem;
  R = 0.;
  Rp = 0.;
  Rpp = 0.;

  for(int i = 0; i < N; i++) {
    double mu = mu_vec(i);
    V_eta(mu, eta, V, Vp, Vpp);
    double Z = 0.5 * pow(data.Y(i) - mu, 2) / V / phi;
    R += omega(i) * (-0.5 * log(phi * V) - Z);
    Rp += omega(i) * (-0.5 / V + Z / V) * Vp;
    Rpp += omega(i) * (Vpp * (-0.5 / V + Z / V)
                       + Vp * Vp * (0.5 / V / V - 2 * Z / V / V));
  }
}

void newton_rhapson(double& eta, double& phi,
                    const arma::vec& omega,
                    const QNBData& data) {
  
  int NUM_NEWTON = 10;
  int N = omega.n_elem;
  double R, Rp, Rpp;
  double V, Vp, Vpp;
  arma::vec mu = exp(data.lambda_hat);
  arma::vec Z = zeros<vec>(N);
  
  for(int k = 0; k < NUM_NEWTON; k++) {

    // Step 1: Coordinate Ascent on phi
    for(int i = 0; i < N; i++) {
      V_eta(mu(i), eta, V, Vp, Vpp);
      Z(i) = pow(data.Y(i) - mu(i), 2) / V;
    }
    phi = sum(Z % omega);

    // Step 2: Newton Step on eta
    R_eta(eta, phi, mu, omega, data, R, Rp, Rpp);
    eta = eta - Rp / Rpp;
  }
}

arma::vec PredictPois(std::vector<QNBNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

void UpdateHypers(QNBParams& hypers, std::vector<QNBNode*>& trees,
                  const QNBData& data)
{

  // UPDATE THIS!!!
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

  // Update the phi and eta by coordinate ascent/Newton's method
  double phi = hypers.phi;
  double k = hypers.k;
  newton_rhapson(k, phi, omega, data);
  hypers.phi = phi;
  hypers.k = k;

  // Update xi
  for(int i = 0; i < N; i++) {
    double a = (k + data.Y(i)) / hypers.phi;
    double b = (k + mu(i)) / hypers.phi;
    xi(i) = R::rgamma(a, 1/b);
  }
}

/****************************************/

// [[Rcpp::export]]
List QNBBart(const arma::mat& X,
              const arma::vec& Y,
              const arma::mat& X_test,
              const arma::sp_mat& probs,
              int num_trees,
              double scale_lambda,
              double scale_lambda_0,
              int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  QNBParams pois_params(scale_lambda_0, scale_lambda, 1.0, 0.01);
  QNBForest forest(num_trees, &tree_hypers, &pois_params);
  QNBData data(X,Y);
  mat lambda = zeros<mat>(num_save, Y.size());
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec phi = zeros<vec>(num_save);
  vec k = zeros<vec>(num_save);

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
    k(iter) = pois_params.get_k();
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;
  out["phi"] = phi;
  out["k"] = k;

  return out;
}
