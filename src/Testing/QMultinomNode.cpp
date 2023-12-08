#include "QBinomNode.h"

using namespace arma;
using namespace Rcpp;

void QBinomNode::AddSuffStat(const QBinomData& data, int i, double phi) {
  ss.Increment(data.Y(i), data.rho(i), data.lambda_hat(i), phi);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, phi);
    } else {
      right->AddSuffStat(data,i, phi);
    }
  }
}

void QBinomNode::UpdateSuffStat(const QBinomData& data, double phi) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, phi);
  }
}

double PredictPois(QBinomNode* n, const rowvec& x) {
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

arma::vec PredictPois(QBinomNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(QBinomData& data, QBinomNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(QBinomData& data, QBinomNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(QBinomNode* root, const QBinomData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QBinomNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    double sum_Y_by_phi = leafs[i]->ss.sum_Y_by_phi;
    double sum_exp_lambda_minus_by_phi = leafs[i]->ss.sum_exp_lambda_minus_by_phi;
    
    out += poisson_lgamma_marginal_loglik(sum_Y_by_phi,
                                          0.,
                                          sum_exp_lambda_minus_by_phi,
                                          root->pois_params->get_alpha(),
                                          root->pois_params->get_beta());
  }
  return out;
}

void UpdateParams(QBinomNode* root, const QBinomData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QBinomNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  for(int i = 0; i < num_leaves; i++) {
    double sum_Y_by_phi = leafs[i]->ss.sum_Y_by_phi;
    double sum_exp_lambda_minus_by_phi = leafs[i]->ss.sum_exp_lambda_minus_by_phi;
    leafs[i]->lambda =
      poisson_lgamma_draw_posterior(sum_Y_by_phi,
                                    0.,
                                    sum_exp_lambda_minus_by_phi,
                                    root->pois_params->get_alpha(),
                                    root->pois_params->get_beta());
  }
}
