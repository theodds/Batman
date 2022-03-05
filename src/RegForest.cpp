#include "RegForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictReg(std::vector<RegNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictReg(forest[n], X);
  }
  return out;
}


void UpdateHypers(RegParams& hypers, std::vector<RegNode*>& trees, const RegData& data) {
  UpdateSigmaY(hypers, data); 
  UpdateSigmaMu(hypers, trees);
}

void get_params(RegNode* n,
                std::vector<double>& mu)
{
  if(n->is_leaf) {
    mu.push_back(n->mu);
  }
  else {
    get_params(n->left, mu);
    get_params(n->right, mu);
  }
}

arma::vec get_params(std::vector<RegNode*>& forest) {
  std::vector<double> mu;
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_params(forest[t], mu);
  }
  int num_leaves = mu.size();
  vec mu_out = zeros<vec>(num_leaves);
  for(int t = 0; t < num_leaves; t++) mu_out(t) = mu[t];
  
  return mu_out;
}

// TODO: Double check this!!!
void UpdateSigmaY(RegParams& hypers, const RegData& data) {
  vec res = data.Y - data.mu_hat; 
  double tau_new = half_cauchy_update_precision_mh(res, hypers.get_prec_y(), 
                                                   hypers.scale_sigma); 
  hypers.set_prec_y(tau_new);
}

void UpdateSigmaMu(RegParams& hypers, std::vector<RegNode*>& forest) {
  vec mu = get_params(forest);
  double tau_new = half_cauchy_update_precision_mh(mu, hypers.get_prec_mu(), 
                                                   hypers.scale_sigma_mu); 
  hypers.set_prec_mu(tau_new); 
}

// [[Rcpp::export]]
List RegBart(const arma::mat& X,
             const arma::vec& Y,
             const arma::mat& X_test,
             const arma::sp_mat& probs,
             int num_trees,
             double scale_sigma,
             double scale_sigma_mu,
             int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs); 
  RegParams reg_params(scale_sigma, scale_sigma_mu,
                       scale_sigma, scale_sigma_mu); 
  RegForest forest(num_trees, &tree_hypers, &reg_params); 
  RegData data(X, Y);
  mat mu = zeros<mat>(num_save, Y.size());
  mat mu_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, reg_params, tree_hypers);
    if((iter + 1) % 100 == 0) 
      Rcpp::Rcout << "Finishing warmup " << iter + 1 << "\n";
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, reg_params, tree_hypers); 
    }
    if((iter + 1) % 100 == 0) 
      Rcpp::Rcout << "Finishing save " << iter + 1 << "\n";
    mu.row(iter) = trans(data.mu_hat);
    mu_test.row(iter) = trans(PredictReg(forest.trees, X_test));
    counts.row(iter) = trans(get_var_counts(forest.trees));
  }
  
  List out; 
  out["mu"] = mu;
  out["mu_test"] = mu_test;
  out["counts"] = counts;
  return out;
}

