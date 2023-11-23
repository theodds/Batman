#include "QGammaNode.h"

using namespace arma;
using namespace Rcpp;

void QGammaNode::AddSuffStat(const QGammaData& data, int i, double phi) {
  ss.Increment(data.Y(i), data.lambda_hat(i), phi);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, phi);
    } else {
      right->AddSuffStat(data,i, phi);
    }
  }
}

void QGammaNode::UpdateSuffStat(const QGammaData& data, double phi) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, phi);
  }
}

double PredictPois(QGammaNode* n, const rowvec& x) {
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

arma::vec PredictPois(QGammaNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(QGammaData& data, QGammaNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(QGammaData& data, QGammaNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(QGammaNode* root, const QGammaData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QGammaNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    double sum_eta = leafs[i]->ss.sum_lambda_minus;
    double sum_y_exp_eta = leafs[i]->ss.sum_exp_lambda_minus_y;
    double alpha = root->pois_params->get_alpha();
    double beta = root->pois_params->get_beta();
    double alpha_up = alpha + sum_eta;
    double beta_up = beta + sum_y_exp_eta;

    out += beta * log(alpha) - R::lgammafn(alpha);
    out += R::lgammafn(alpha_up) - beta_up * log(alpha_up);
  }
  return out;
}

void UpdateParams(QGammaNode* root, const QGammaData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QGammaNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  for(int i = 0; i < num_leaves; i++) {
    double sum_eta = leafs[i]->ss.sum_lambda_minus;
    double sum_y_exp_eta = leafs[i]->ss.sum_exp_lambda_minus_y;
    double alpha = root->pois_params->get_alpha();
    double beta = root->pois_params->get_beta();
    double alpha_up = alpha + sum_eta;
    double beta_up = beta + sum_y_exp_eta;

    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}
