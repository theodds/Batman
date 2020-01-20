#include "RegNode.h"

using namespace arma;
using namespace Rcpp;

void RegNode::AddSuffStat(const RegData& data, int i) {
  double Z = data.Y(i) - data.mu_hat(i);
  ss.Increment(Z);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i);
    } else {
      right->AddSuffStat(data, i);
    }
  }
}

void RegNode::UpdateSuffStat(const RegData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data, i);
  }
}


double PredictReg(RegNode* n, const rowvec& x) {
  if(n->is_leaf) {
    return n->mu;
  }
  if(x(n->var) <= n->val) {
    return PredictReg(n->left, x);
  }
  else {
    return PredictReg(n->right, x);
  }
}

arma::vec PredictReg(RegNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictReg(tree, x);
  }
  return out;
}

void BackFit(RegData& data, RegNode* tree) {
  vec mu = PredictReg(tree, data.X);
  data.mu_hat = data.mu_hat - mu;
}

void Refit(RegData& data, RegNode* tree) {
  vec mu = PredictReg(tree, data.X);
  data.mu_hat = data.mu_hat + mu;
}

// [[Rcpp::export]]
void doit() {
  sp_mat probs = zeros<sp_mat>(3,2);
  probs(0,0) = 1.0;
  probs(1,1) = 0.5;
  probs(2,1) = 0.5;
  TreeHypers* tree_hypers = new TreeHypers(probs);
  RegParams* reg_params = new RegParams(1.0,1.0,1.0,1.0);

  RegNode* x = new RegNode(tree_hypers, reg_params);
  Rcout << x->mu << std::endl;
  x->BirthLeaves();
  Rcout << x->left->tree_hypers->gamma << std::endl;
  Rcout << "(" << x->var << "," << x->val << ")" << std::endl;

  Rcout << "SAMPLING!!" << std::endl;
  uvec counts = zeros<uvec>(3);
  for(int i = 0; i < 100; i++) {
    int j = tree_hypers->SampleVar()(1);
    counts(j) = counts(j) + 1;
  }
  Rcout << counts << std::endl;

  x->left->ss.Increment(3.0);
  Rcout << x->left->ss.sum_Y_2;
  ResetSuffStat(x);
  Rcout << x->left->ss.sum_Y_2;

  delete x;
  delete tree_hypers;
  delete reg_params;

}

double LogLT(RegNode* root, const RegData& data) {
  root->UpdateSuffStat(data);
  std::vector<RegNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    RegSuffStats* rss = &(leafs[i]->ss);
    out += gaussian_gaussian_marginal_loglik(rss->sum_Y_0,
                                             rss->sum_Y_1,
                                             rss->sum_Y_2,
                                             root->reg_params->get_prec_y(),
                                             root->reg_params->get_prec_mu());
  }

  return out;

}

void UpdateParams(RegNode* root, const RegData& data) {

  root->UpdateSuffStat(data);
  std::vector<RegNode*> leafs = leaves(root);
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    RegSuffStats* rss = &(leafs[i]->ss);
    leafs[i]->mu =
      gaussian_gaussian_draw_posterior(rss->sum_Y_0,
                                       rss->sum_Y_1,
                                       root->reg_params->get_prec_y(),
                                       root->reg_params->get_prec_mu());
  }
}
