#include "PoisNode.h"

using namespace arma;
using namespace Rcpp;

void PoisNode::AddSuffStat(const PoisData& data, int i) {
  ss.Increment(data.Y(i), data.lambda_hat(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i);
    } else {
      right->AddSuffStat(data,i);
    }
  }
}

void PoisNode::UpdateSuffStat(const PoisData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i);
  }
}

double PredictPois(PoisNode* n, const rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return PredictPois(n->left, x);
  }
  else {
    return PredictPois(n->right, x);
  }
}

arma::vec PredictPois(PoisNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(PoisData& data, PoisNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(PoisData& data, PoisNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(PoisNode* root, const PoisData& data) {
  root->UpdateSuffStat(data);
  std::vector<PoisNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    double sum_Y = leafs[i]->ss.sum_Y;
    double sum_Y_lambda_minus = leafs[i]->ss.sum_Y_lambda_minus;
    double sum_exp_lambda_minus = leafs[i]->ss.sum_exp_lambda_minus;

    out += poisson_lgamma_marginal_loglik(sum_Y,
                                          sum_Y_lambda_minus,
                                          sum_exp_lambda_minus,
                                          root->pois_params->get_alpha(),
                                          root->pois_params->get_beta());
  }
  return out;
}

void UpdateParams(PoisNode* root, const PoisData& data) {
  root->UpdateSuffStat(data);
  std::vector<PoisNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  for(int i = 0; i < num_leaves; i++) {
    double sum_Y = leafs[i]->ss.sum_Y;
    double sum_Y_lambda_minus = leafs[i]->ss.sum_Y_lambda_minus;
    double sum_exp_lambda_minus = leafs[i]->ss.sum_exp_lambda_minus;
    leafs[i]->lambda =
      poisson_lgamma_draw_posterior(sum_Y,
                                    sum_Y_lambda_minus,
                                    sum_exp_lambda_minus,
                                    root->pois_params->get_alpha(),
                                    root->pois_params->get_beta());
  }
}
