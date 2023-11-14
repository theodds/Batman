#include "CoxPENode.h"

using namespace arma;
using namespace Rcpp;

void CoxPENode::AddSuffStat(const CoxPEData& data, int i)
{
  ss.Increment(data.delta(i), data.lambda_hat(i), data.cum_base_haz_Y(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i);
    } else {
      right->AddSuffStat(data, i);
    }
  }
}

void CoxPENode::UpdateSuffStat(const CoxPEData& data)
{
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data, i);
  }
}

double PredictCox(CoxPENode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return PredictCox(n->left, x);
  }
  else {
    return PredictCox(n->right, x);
  }
}

arma::vec PredictCox(CoxPENode* tree, const arma::mat& X)
{
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictCox(tree, x);
  }
  return out;
}

void BackFit(CoxPEData& data, CoxPENode* tree) {
  vec lambda = PredictCox(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(CoxPEData& data, CoxPENode* tree) {
  vec lambda = PredictCox(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(CoxPENode* root, const CoxPEData& data) {
  root->UpdateSuffStat(data);
  std::vector<CoxPENode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();
  double lgamma_alpha = R::lgammafn(alpha);
  double alpha_log_beta = alpha * log(beta);

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.sum_delta;
    double beta_up  = beta + leafs[i]->ss.sum_exp_lambda_times_cum_base_haz;

    out += alpha_log_beta - lgamma_alpha
      - alpha_up * log(beta_up)  + R::lgammafn(alpha_up);
    // out += exp(leafs[i]->ss.sum_delta_lambda_minus);
  }

  return out;

}

void UpdateParams(CoxPENode* root, const CoxPEData& data) {
  root->UpdateSuffStat(data);
  std::vector<CoxPENode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.sum_delta;
    double beta_up  = beta + leafs[i]->ss.sum_exp_lambda_times_cum_base_haz;
    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}
