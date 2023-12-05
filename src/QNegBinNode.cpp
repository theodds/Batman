#include "QNegBinNode.h"

using namespace arma;
using namespace Rcpp;

void QNBNode::AddSuffStat(const QNBData& data, int i, double phi) {
  ss.Increment(data.Y(i), data.lambda_hat(i), phi, data.xi(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, phi);
    } else {
      right->AddSuffStat(data,i, phi);
    }
  }
}

void QNBNode::UpdateSuffStat(const QNBData& data, double phi) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, phi);
  }
}

double PredictPois(QNBNode* n, const rowvec& x) {
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

arma::vec PredictPois(QNBNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(QNBData& data, QNBNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(QNBData& data, QNBNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(QNBNode* root, const QNBData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QNBNode*> leafs = leaves(root);

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

void UpdateParams(QNBNode* root, const QNBData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QNBNode*> leafs = leaves(root);
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
