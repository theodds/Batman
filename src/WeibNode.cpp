#include "WeibNode.h"

using namespace arma;
using namespace Rcpp;

void WeibNode::AddSuffStat(const rowvec& x,
                           double y_elam,
                           double num_w,
                           double sum_log_w,
                           double lam_num_w) {

  ss.Increment(y_elam, num_w, sum_log_w, lam_num_w);
  if(!is_leaf) {
    if(x(var) <= val) {
      left->AddSuffStat(x, y_elam, num_w, sum_log_w, lam_num_w);
    }
    else {
      right->AddSuffStat(x, y_elam, num_w, sum_log_w, lam_num_w);
    }
  }
}

void WeibNode::UpdateSuffStat(const WeibData& data, double weibull_power) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    double y_elam = pow(data.Y(i), weibull_power) * exp(data.mu_hat(i));
    double sum_log_w = 0.;
    for(int j = 0; j < data.W[i].size(); j++) {
      sum_log_w += log(data.W[i][j]);
    }
    double lam_num_w = data.mu_hat(i) * data.W[i].size();
    rowvec x = data.X.row(i);
    AddSuffStat(x, y_elam, data.W[i].size(), sum_log_w, lam_num_w);
  }
}

double PredictWeib(WeibNode* n, const arma::rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return PredictWeib(n->left, x);
  } else {
    return PredictWeib(n->right, x);
  }
}


arma::vec PredictWeib(WeibNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictWeib(tree,x);
  }
  return out;
}

void BackFit(WeibData& data, WeibNode* tree) {
  vec lambda = PredictWeib(tree, data.X);
  data.mu_hat = data.mu_hat - lambda;
}

void Refit(WeibData& data, WeibNode* tree) {
  vec lambda = PredictWeib(tree, data.X);
  data.mu_hat = data.mu_hat + lambda;
}

double LogLT(WeibNode* root, const WeibData& data) {
  root->UpdateSuffStat(data, root->weib_params->weibull_power);
  std::vector<WeibNode*> leafs = leaves(root);

  double out = 0.;
  int num_leaves = leafs.size();

  double alpha = root->weib_params->get_alpha();
  double beta = root->weib_params->get_beta();
  double alpha_log_beta = alpha * log(beta);
  double kappa = root->weib_params->weibull_power;
  double lgamma_alpha = R::lgammafn(alpha);
  double weibull_power = root->weib_params->weibull_power;

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.num_W;
    double beta_up = beta + leafs[i]->ss.sum_Y_elam;
    out += leafs[i]->ss.num_W * log(weibull_power);
    out += alpha_log_beta - lgamma_alpha;
    out += R::lgammafn(alpha_up) - alpha_up * log(beta_up);
    out += (weibull_power - 1) * leafs[i]->ss.sum_log_W;
    out += leafs[i]->ss.sum_lam_num_W;
    
  }
  return out;
}

void UpdateParams(WeibNode* root, const WeibData& data) {
  root->UpdateSuffStat(data, root->weib_params->weibull_power);
  std::vector<WeibNode*> leafs = leaves(root);
  int num_leaves = leafs.size();

  double alpha = root->weib_params->get_alpha();
  double beta = root->weib_params->get_beta();
  double alpha_log_beta = alpha * log(beta);
  double kappa = root->weib_params->weibull_power;
  double lgamma_alpha = R::lgammafn(alpha);

  for(int i = 0; i < num_leaves; i++) {
    double alpha_up = alpha + leafs[i]->ss.num_W;
    double beta_up = beta + leafs[i]->ss.sum_Y_elam;
    leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
  }
}


