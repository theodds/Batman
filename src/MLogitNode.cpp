#include "MLogitNode.h"

using namespace arma;
using namespace Rcpp;

void MLogitNode::AddSuffStat(const MLogitData& data,
                             int i,
                             const arma::vec& exp_lambda_hat) {
  int k = data.Y(i);
  ss.Increment(k, exp_lambda_hat, data.phi(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, exp_lambda_hat);
    } else {
      right->AddSuffStat(data, i, exp_lambda_hat);
    }
  }
}

void MLogitNode::UpdateSuffStat(const MLogitData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    vec exp_lambda_hat = trans(exp(data.lambda_hat.row(i)));
    AddSuffStat(data, i, exp_lambda_hat);
  }
}

arma::vec PredictMLogit(MLogitNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return PredictMLogit(n->left, x);
  } else {
    return PredictMLogit(n->right,x);
  }
}

arma::mat PredictMLogit(MLogitNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N,tree->lambda.size());
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = trans(PredictMLogit(tree, x));
  }
  return out;
}

void BackFit(MLogitData& data, MLogitNode* tree) {
  mat lambda = PredictMLogit(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(MLogitData& data, MLogitNode* tree) {
  mat lambda = PredictMLogit(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(MLogitNode* root, const MLogitData& data) {
  root->UpdateSuffStat(data);
  std::vector<MLogitNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();
  int num_cat = root->lambda.size();

  double alpha = root->mlogit_params->get_alpha();
  double beta = root->mlogit_params->get_beta();
  double alpha_log_beta = alpha * log(beta);
  double lgamma_alpha = R::lgammafn(alpha);

  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < num_cat; k++) {
      double alpha_up = alpha + leafs[i]->ss.sum_Y(k);
      double beta_up  = beta + leafs[i]->ss.sum_exp_lambda_minus_phi(k);
      out += alpha_log_beta - lgamma_alpha;
      out += R::lgammafn(alpha_up) - alpha_up * log(beta_up);
    }
  }
  return out;
}

// TODO: Combine update params into evaluation of log likelihood???
void UpdateParams(MLogitNode* root, const MLogitData& data) {
  root->UpdateSuffStat(data);
  std::vector<MLogitNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  int num_cat = root->lambda.size();

  double alpha = root->mlogit_params->get_alpha();
  double beta = root->mlogit_params->get_beta();
  double alpha_log_beta = alpha * log(beta);
  double lgamma_alpha = R::lgammafn(alpha);

  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < num_cat; k++) {
      double alpha_up = alpha + leafs[i]->ss.sum_Y(k);
      double beta_up  = beta + leafs[i]->ss.sum_exp_lambda_minus_phi(k);
      leafs[i]->lambda(k) =
        rlgam(alpha_up) - log(beta_up);
    }
  }
}

