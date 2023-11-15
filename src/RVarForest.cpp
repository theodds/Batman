#include "RVarForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictVar(std::vector<RVarNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec tau = ones<vec>(X.n_rows);
  for(int n = 0; n < N; n++) {
    tau = tau % PredictVar(forest[n], X);
  }
  return tau;
}

void get_params(RVarNode* n,, std::vector<double>& tau) {
  if(n->is_leaf) {
    tau.push_back(n->tau);
  }
  else {
    get_params(n->left, tau);
    get_params(n->right, tau);
  }
}

void UpdateHypers(RVarParams& var_params,
                  std::vector<RVarNode*>& trees,
                  RVarData& data)
{
  std::vector<double> tau;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], tau);
  }
  // Rcout << "Update Tau";
  UpdateTau0(var_params, data);
  // Rcout << "Update Scale";
  UpdateScaleLogTau(var_params, tau);
  // Rcout << "Done";
}

void UpdateTau0(RVarParams& var_params, RVarData& data) {
  int N = data.Y.size();
  double shape_up = var_params.shape_tau_0 + 0.5 * N;
  double rate_up = var_params.rate_tau_0;
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) / var_params.tau_0;
    rate_up += 0.5 * data.tau_hat(i) * pow(data.Y(i), 2.0);
  }
  double scale_up = 1.0 / rate_up;
  var_params.tau_0 = R::rgamma(shape_up, scale_up);
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) * var_params.tau_0;
  }
}

void UpdateScaleLogTau(RVarParams& var_params, std::vector<double>& tau) {
  double n = (double)tau.size();
  double sum_lambda = 0.;
  double sum_exp_lambda = 0.;
  for(int i = 0; i < tau.size(); i++) {
    sum_lambda += log(tau[i]);
    sum_exp_lambda += tau[i];
  }
  double scale = var_params.sigma_scale_log_tau;
  ScaleLambdaLoglik* loglik =
    new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  double scale_0 = var_params.get_scale_log_tau();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  var_params.set_scale_log_tau(scale_1);

  delete loglik;
}

// [[Rcpp::export]]
List RVarBart(const arma::mat& X,
             const arma::vec& Y,
             const arma::sp_mat& probs,
             double sigma_scale_log_tau,
             double shape_tau_0, double rate_tau_0,
             int num_trees,
             int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  RVarParams var_params(sigma_scale_log_tau,
                        sigma_scale_log_tau,
                        shape_tau_0,
                        rate_tau_0);
  RVarForest forest(num_trees, &tree_hypers, &var_params);
  RVarData data(X, Y);
  mat tau = zeros<mat>(num_save, Y.size());
  umat counts = zeros<umat>(num_save, probs.n_cols);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, var_params, tree_hypers);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, var_params, tree_hypers);
    }
    tau.row(iter) = trans(data.tau_hat);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }
  List out;
  out["tau"] = tau;
  out["counts"] = counts;
  
  return out;
}

