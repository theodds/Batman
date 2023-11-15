#include "RVarNode.h"

using namespace arma;
using namespace Rcpp;

void RVarNode::AddSuffStat(const RVarData& data, int i) {
  ss.Increment(data.Y(i), data.tau_hat(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i);
    } else {
      right->AddSuffStat(data, i);
    }
  }
}

void RVarNode::UpdateSuffStat(const RVarData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data, i);
  }
}

double PredictVar(RVarNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    return n->tau;
  }
  if(x(n->var) <= n->val) {
    return PredictVar(n->left, x);
  } else {
    return PredictVar(n->right,x);
  }
}

arma::vec PredictVar(RVarNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictVar(tree, x);
  }
  return out;
}

void BackFit(RVarData& data, RVarNode* tree) {
  vec tau = PredictVar(tree, data.X);
  data.tau_hat = data.tau_hat / tau;
}

void Refit(RVarData& data, RVarNode* tree) {
  vec tau = PredictVar(tree, data.X);
  data.tau_hat = data.tau_hat % tau;
}

double LogLT(RVarNode* root, const RVarData& data) {
  root->UpdateSuffStat(data);
  std::vector<RVarNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();

  double alpha = root->var_params->get_alpha();
  double beta  = root->var_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    RVarSuffStats* ss = &(leafs[i]->ss);
    out += weighted_normal_mean0_gamma_loglik(ss->n,           ss->sum_eta,
                                              ss->sum_eta_y,   ss->sum_eta_y_sq,
                                              ss->sum_log_eta, alpha,
                                              beta);
  }
  return out;
}

void UpdateParams(RVarNode* root, const RVarData& data) {
  root->UpdateSuffStat(data);
  std::vector<RVarNode*> leafs = leaves(root);
  int num_leaves = leafs.size();

  double alpha = root->var_params->get_alpha();
  double beta  = root->var_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    RVarSuffStats* ss = &(leafs[i]->ss);
    leafs[i]->tau = weighted_normal_mean0_gamma_draw(ss->n,
                                                     ss->sum_eta,
                                                     ss->sum_eta_y,
                                                     ss->sum_eta_y_sq,
                                                     ss->sum_log_eta,
                                                     alpha,
                                                     beta);
  }
}
