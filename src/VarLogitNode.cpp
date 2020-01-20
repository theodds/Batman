#include "VarLogitNode.h"

using namespace arma;
using namespace Rcpp;


void VarLogitNode::AddSuffStatLogit(const MLogitData& data, int i,
                                    const arma::vec& exp_lambda_hat)
{
  int k = data.Y(i);
  ss.IncrementLogit(k, exp_lambda_hat, data.phi(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStatLogit(data, i, exp_lambda_hat);
    } else {
      right->AddSuffStatLogit(data, i, exp_lambda_hat);
    }
  }
}

void VarLogitNode::AddSuffStatVar(const VarData& data, int i) {
  ss.IncrementVar(data.Y(i) - data.mu_hat(i), data.tau_hat(i));
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStatVar(data, i);
    } else {
      right->AddSuffStatVar(data,i);
    }
  }
}

void VarLogitNode::UpdateSuffStat(const VarData& data_var,
                                  const MLogitData& data_logit) {
  
  ResetSuffStat();
  for(int i = 0; i < data_var.X.n_rows; i++) {
    AddSuffStatVar(data_var, i);
  }
  for(int i = 0; i < data_logit.X.n_rows; i++) {
    vec exp_lambda_hat = trans(exp(data_logit.lambda_hat.row(i)));
    AddSuffStatLogit(data_logit, i, exp_lambda_hat);
  }
}

arma::vec PredictMLogit(VarLogitNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return PredictMLogit(n->left, x);
  } else {
    return PredictMLogit(n->right,x);
  }
}

arma::mat PredictMLogit(VarLogitNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N,tree->lambda.size());
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = trans(PredictMLogit(tree, x));
  }
  return out;
}

arma::vec PredictVar(VarLogitNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    vec out = zeros<vec>(2);
    out(0) = n->mu;
    out(1) = n->tau;
    return out;
  }
  if(x(n->var) <= n->val) {
    return PredictVar(n->left, x);
  } else {
    return PredictVar(n->right,x);
  }
}

arma::mat PredictVar(VarLogitNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N,2);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = trans(PredictVar(tree, x));
  }
  return out;
}

void BackFit(VarLogitData& data, VarLogitNode* tree) {

  // Backfit regression
  mat mu_tau = PredictVar(tree, data.var_data.X);
  data.var_data.mu_hat =
    data.var_data.mu_hat - mu_tau.col(0);
  data.var_data.tau_hat =
    data.var_data.tau_hat / mu_tau.col(1);

  // Backfit logit
  mat lambda = PredictMLogit(tree, data.mlogit_data.X);
  data.mlogit_data.lambda_hat = data.mlogit_data.lambda_hat - lambda;
}

void Refit(VarLogitData& data, VarLogitNode* tree) {
  // Refit regression
  mat mu_tau = PredictVar(tree, data.var_data.X);
  data.var_data.mu_hat =
    data.var_data.mu_hat + mu_tau.col(0);
  data.var_data.tau_hat =
    data.var_data.tau_hat % mu_tau.col(1);

  // Refit logit
  mat lambda = PredictMLogit(tree, data.mlogit_data.X);
  data.mlogit_data.lambda_hat = data.mlogit_data.lambda_hat + lambda;

}

double LogLT(VarLogitNode* root, const VarLogitData& data) {
  root->UpdateSuffStat(data.var_data, data.mlogit_data);
  std::vector<VarLogitNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();
  int num_cat = root->lambda.size();

  double alpha_reg = root->params->var_params.get_alpha();
  double beta_reg  = root->params->var_params.get_beta();
  double alpha_cat = root->params->mlogit_params.get_alpha();
  double beta_cat = root->params->mlogit_params.get_beta();
  double alpha_log_beta = alpha_cat * log(beta_cat);
  double lgamma_alpha = R::lgammafn(alpha_cat);

  for(int i = 0; i < num_leaves; i++) {
    VarSuffStats* ss = &(leafs[i]->ss.var_stats);
    out += weighted_normal_gamma_loglik(ss->n          , ss->sum_eta,
                                        ss->sum_eta_y  , ss->sum_eta_y_sq,
                                        ss->sum_log_eta, alpha_reg,
                                        beta_reg       , root->params->var_params.kappa);
  }

  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < num_cat; k++) {
      double alpha_up = alpha_cat + leafs[i]->ss.mlogit_stats.sum_Y(k);
      double beta_up  = beta_cat +
        leafs[i]->ss.mlogit_stats.sum_exp_lambda_minus_phi(k);
      out += alpha_log_beta - lgamma_alpha;
      out += R::lgammafn(alpha_up) - alpha_up * log(beta_up);
    }
  }
  return out;
}

void UpdateParams(VarLogitNode* root, const VarLogitData& data) {
  root->UpdateSuffStat(data.var_data, data.mlogit_data);
  std::vector<VarLogitNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();
  int num_cat = root->lambda.size();

  double alpha_reg = root->params->var_params.get_alpha();
  double beta_reg  = root->params->var_params.get_beta();
  double alpha_cat = root->params->mlogit_params.get_alpha();
  double beta_cat = root->params->mlogit_params.get_beta();

  for(int i = 0; i < num_leaves; i++) {
    VarSuffStats* ss = &(leafs[i]->ss.var_stats);
    vec mu_tau =
      weighted_normal_gamma_draw_posterior(ss->n, ss->sum_eta, ss->sum_eta_y,
                                           ss->sum_eta_y_sq, ss->sum_log_eta,
                                           alpha_reg, beta_reg,
                                           root->params->var_params.kappa);
    leafs[i]->mu = mu_tau(0);
    leafs[i]->tau = mu_tau(1);
  }

  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < num_cat; k++) {
      double alpha_up = alpha_cat + leafs[i]->ss.mlogit_stats.sum_Y(k);
      double beta_up  = beta_cat +
        leafs[i]->ss.mlogit_stats.sum_exp_lambda_minus_phi(k);
      leafs[i]->lambda(k) = rlgam(alpha_up) - log(beta_up);
    }
  }
}


