#include "CLogLogOrdinalNode.h"

using namespace arma;
using namespace Rcpp;

void CLogLogOrdinalNode::AddSuffStat(const CLogLogOrdinalData& data,
                                     int i,
                                     const arma::vec& gamma,
                                     const arma::vec& seg) {
  ss.Increment(data.Y(i), data.lambda_hat(i), data.Z(i), gamma, seg);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, gamma, seg);
    } else {
      right->AddSuffStat(data, i, gamma, seg);
    }
  }
}

void CLogLogOrdinalNode::UpdateSuffStat(const CLogLogOrdinalData& data,
                                        const arma::vec& gamma,
                                        const arma::vec& seg) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, gamma, seg);
  }
}

double PredictPois(CLogLogOrdinalNode* n, const rowvec& x) {
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

arma::vec PredictPois(CLogLogOrdinalNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(CLogLogOrdinalData& data, CLogLogOrdinalNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(CLogLogOrdinalData& data, CLogLogOrdinalNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(CLogLogOrdinalNode* root, const CLogLogOrdinalData& data) {
  root->UpdateSuffStat(data, root->pois_params->gamma, root->pois_params->seg);
  std::vector<CLogLogOrdinalNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();

  for(int i = 0; i < num_leaves; i++) {
    double A = leafs[i]->ss.sum_Y_less_K;
    double B = leafs[i]->ss.other_sum;
    double alpha = root->pois_params->get_alpha();
    double beta = root->pois_params->get_beta();
    double alpha_up = alpha + A;
    double beta_up = beta + B;

    out += beta * log(alpha) - R::lgammafn(alpha);
    out += R::lgammafn(alpha_up) - beta_up * log(alpha_up);
  }
  return out;
}

void UpdateParams(CLogLogOrdinalNode* root, const CLogLogOrdinalData& data) {
  root->UpdateSuffStat(data, root->pois_params->gamma, root->pois_params->seg);
  std::vector<CLogLogOrdinalNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  for(int i = 0; i < num_leaves; i++) {
    double A = leafs[i]->ss.sum_Y_less_K;
    double B = leafs[i]->ss.other_sum;
    double alpha = root->pois_params->get_alpha();
    double beta = root->pois_params->get_beta();
    double alpha_up = alpha + A;
    double beta_up = beta + B;

    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}
