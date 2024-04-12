#include "CoxNPHNode.h"

using namespace arma;
using namespace Rcpp;

void CoxNPHNode::AddSuffStat(double delta_b,
                             double lambda_minus,
                             double Z,
                             double base_haz,
                             double b_float,
                             const arma::rowvec& x) {
  ss.Increment(delta_b, lambda_minus, Z, base_haz);
  if(!is_leaf) {
    double xx = (var == x.n_elem - 1) ? b_float : x(var);
    if(xx <= val) {
      left->AddSuffStat(delta_b, lambda_minus, Z, base_haz, b_float, x);
    } else {
      right->AddSuffStat(delta_b, lambda_minus, Z, base_haz, b_float, x);
    }
  }
}

void CoxNPHNode::UpdateSuffStat(const CoxNPHData& data)
{
  ResetSuffStat();
  int N = data.X.n_rows;
  int K = data.base_haz.n_elem;
  for(int i = 0; i < N; i++) {
    rowvec x = data.X.row(i);
    for(int b = 0; b <= data.obs_to_bin(i); b++) {
      double delta_b = data.delta(i) * (data.obs_to_bin(i) == b ? 1. : 0.);
      double b_float = (double)b / (data.base_haz.n_elem - 1.);
      AddSuffStat(delta_b,
                  data.lambda_hat(i,b),
                  data.Z(i,b),
                  data.base_haz(b),
                  b_float,
                  x);
    }
  }
}

double PredictCox(CoxNPHNode* n,
                  const arma::rowvec& x,
                  int num_bin,
                  double b_float) {
  if(n->is_leaf) {
    return n->lambda;
  }
  double xx = (n->var == x.n_elem - 1) ? b_float : x(n->var);
  if(xx < n->val) {
    return PredictCox(n->left, x, num_bin, b_float);
  }
  else {
    return PredictCox(n->right, x, num_bin, b_float);
  }
}

arma::rowvec PredictCox(CoxNPHNode* n, const arma::rowvec& x, int num_bin) {
  rowvec lambda = zeros<rowvec>(num_bin);
  for(int b = 0; b < num_bin; b++) {
    double b_float = (double)b / ((double)num_bin-1);
    lambda(b) = PredictCox(n, x, num_bin, b_float);
  }
  return lambda;
}

arma::mat PredictCox(CoxNPHNode* tree, const arma::mat& X, int num_bin)
{
  int N = X.n_rows;
  mat out = zeros<mat>(N, num_bin);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = PredictCox(tree, x, num_bin);
  }
  return out;
}

void BackFit(CoxNPHData& data, CoxNPHNode* tree) {
  int num_bin = data.Z.n_cols;
  mat lambda = PredictCox(tree, data.X, num_bin);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(CoxNPHData& data, CoxNPHNode* tree) {
  int num_bin = data.Z.n_cols;
  mat lambda = PredictCox(tree, data.X, num_bin);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(CoxNPHNode* root, const CoxNPHData& data) {
  root->UpdateSuffStat(data);
  std::vector<CoxNPHNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();
  double lgamma_alpha = R::lgammafn(alpha);
  double alpha_log_beta = alpha * log(beta);

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.sum_delta;
    double beta_up  = beta + leafs[i]->ss.sum_Z_haz_r;

    out += alpha_log_beta - lgamma_alpha
      - alpha_up * log(beta_up)  + R::lgammafn(alpha_up);
  }

  return out;

}

void UpdateParams(CoxNPHNode* root, const CoxNPHData& data) {
  root->UpdateSuffStat(data);
  std::vector<CoxNPHNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.sum_delta;
    double beta_up  = beta + leafs[i]->ss.sum_Z_haz_r;
    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}
