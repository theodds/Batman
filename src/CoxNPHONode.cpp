#include "CoxNPHONode.h"

using namespace arma;
using namespace Rcpp;

void CoxNPHONode::AddSuffStat(double delta_b,
                             double Z,
                             double lambda_minus,
                             double gamma,
                             double b_float,
                             const arma::rowvec& x) {
  ss.Increment(delta_b, Z, lambda_minus, gamma);
  if(!is_leaf) {
    double xx = (var == x.n_elem - 1) ? b_float : x(var);
    if(xx <= val) {
      left->AddSuffStat(delta_b, Z, lambda_minus, gamma, b_float, x);
    } else {
      right->AddSuffStat(delta_b, Z, lambda_minus, gamma, b_float, x);
    }
  }
}

void CoxNPHONode::UpdateSuffStat(const CoxNPHOData& data, const arma::vec& gamma)
{
  ResetSuffStat();
  int N = data.X.n_rows;
  int K = data.lambda_hat.n_cols + 1;
  for(int i = 0; i < N; i++) {
    rowvec x = data.X.row(i);
    if(data.Y(i) == (K - 1)) {
      for(int k = 0; k < K - 1; k++) {
        double b_float = double(k) / (double(K) - 1);
        AddSuffStat(0., 1., data.lambda_hat(i,k), gamma(k), b_float, x);
      }
    }
    else {
      for(int k = 0; k < data.Y(i); k++) {
        double b_float = double(k) / (double(K) - 1);
        AddSuffStat(0., 1., data.lambda_hat(i,k), gamma(k), b_float, x);
      }
      AddSuffStat(1., data.Z(i,data.Y(i)), data.lambda_hat(i,data.Y(i)), 
                  gamma(data.Y(i)), data.Y(i) / (double(K) - 1), x);
    }
  }
}

double PredictCox(CoxNPHONode* n,
                  const arma::rowvec& x,
                  int K,
                  double b_float) {
  if(n->is_leaf) {
    return n->lambda;
  }
  double xx = (n->var == x.n_elem - 1) ? b_float : x(n->var);
  if(xx < n->val) {
    return PredictCox(n->left, x, K, b_float);
  }
  else {
    return PredictCox(n->right, x, K, b_float);
  }
}

arma::rowvec PredictCox(CoxNPHONode* n, const arma::rowvec& x, int K) {
  rowvec lambda = zeros<rowvec>(K - 1);
  for(int b = 0; b < K - 1; b++) {
    double b_float = (double)b / ((double)K-1);
    lambda(b) = PredictCox(n, x, K, b_float);
  }
  return lambda;
}

arma::mat PredictCox(CoxNPHONode* tree, const arma::mat& X, int K)
{
  int N = X.n_rows;
  mat out = zeros<mat>(N, K - 1);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = PredictCox(tree, x, K);
  }
  return out;
}

void BackFit(CoxNPHOData& data, CoxNPHONode* tree) {
  int num_bin = data.Z.n_cols + 1;
  mat lambda = PredictCox(tree, data.X, num_bin);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(CoxNPHOData& data, CoxNPHONode* tree) {
  int num_bin = data.Z.n_cols + 1;
  mat lambda = PredictCox(tree, data.X, num_bin);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(CoxNPHONode* root, const CoxNPHOData& data) {
  root->UpdateSuffStat(data, root->cox_params->gamma);
  std::vector<CoxNPHONode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();
  double lgamma_alpha = R::lgammafn(alpha);
  double alpha_log_beta = alpha * log(beta);

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.A_l;
    double beta_up  = beta + leafs[i]->ss.B_l;

    out += alpha_log_beta - lgamma_alpha
      - alpha_up * log(beta_up)  + R::lgammafn(alpha_up);
  }

  return out;

}

void UpdateParams(CoxNPHONode* root, const CoxNPHOData& data) {
  root->UpdateSuffStat(data, root->cox_params->gamma);
  std::vector<CoxNPHONode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  double alpha = root->cox_params->get_alpha();
  double beta = root->cox_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.A_l;
    double beta_up  = beta + leafs[i]->ss.B_l;
    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}
