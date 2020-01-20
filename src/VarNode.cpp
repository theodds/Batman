#include "VarNode.h"

using namespace arma;
using namespace Rcpp;

void VarNode::AddSuffStat(const VarData& data, int i) {
  ss.Increment(data.Y(i) - data.mu_hat(i), data.tau_hat(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i);
    } else {
      right->AddSuffStat(data, i);
    }
  }
}

void VarNode::UpdateSuffStat(const VarData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data, i);
  }
}

arma::vec PredictVar(VarNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    vec out = zeros<vec>(2);
    out(0) = n->mu;
    out(1) = n->tau;
    return out;
  }
  if(x(n->var) <= n->val) {
    return PredictVar(n->left, x);
  } else {
    return PredictVar(n->right,x);
  }
}

arma::mat PredictVar(VarNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N,2);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = trans(PredictVar(tree, x));
  }
  return out;
}

void BackFit(VarData& data, VarNode* tree) {
  mat mu_tau = PredictVar(tree, data.X);
  data.mu_hat = data.mu_hat - mu_tau.col(0);
  data.tau_hat = data.tau_hat / mu_tau.col(1);
}

void Refit(VarData& data, VarNode* tree) {
  mat mu_tau = PredictVar(tree, data.X);
  data.mu_hat = data.mu_hat + mu_tau.col(0);
  data.tau_hat = data.tau_hat % mu_tau.col(1);
}

double LogLT(VarNode* root, const VarData& data) {
  root->UpdateSuffStat(data);
  std::vector<VarNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();

  double alpha = root->var_params->get_alpha();
  double beta  = root->var_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    VarSuffStats* ss = &(leafs[i]->ss);
    out += weighted_normal_gamma_loglik(ss->n,           ss->sum_eta,
                                        ss->sum_eta_y,   ss->sum_eta_y_sq,
                                        ss->sum_log_eta, alpha,
                                        beta,            root->var_params->kappa);
  }
  return out;
}

void UpdateParams(VarNode* root, const VarData& data) {
  root->UpdateSuffStat(data);
  std::vector<VarNode*> leafs = leaves(root);
  int num_leaves = leafs.size();

  double alpha = root->var_params->get_alpha();
  double beta  = root->var_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    VarSuffStats* ss = &(leafs[i]->ss);
    vec mu_tau =
      weighted_normal_gamma_draw_posterior(ss->n, ss->sum_eta,
                                           ss->sum_eta_y, ss->sum_eta_y_sq,
                                           ss->sum_log_eta, alpha,
                                           beta, root->var_params->kappa);
    leafs[i]->mu = mu_tau(0);
    leafs[i]->tau = mu_tau(1);
  }
}
