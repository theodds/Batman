#include "QPowerNode.h"

using namespace arma;
using namespace Rcpp;

void QPowerNode::AddSuffStat(const QPowerData& data, int i, double phi, double p) {
  ss.Increment(data.Y(i), data.lambda_hat(i), phi, p);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, phi, p);
    } else {
      right->AddSuffStat(data,i, phi, p);
    }
  }
}

void QPowerNode::UpdateSuffStat(const QPowerData& data, double phi, double p) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, phi, p);
  }
}

double PredictPois(QPowerNode* n, const rowvec& x) {
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

arma::vec PredictPois(QPowerNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out(i) = PredictPois(tree, x);
  }
  return out;
}

void BackFit(QPowerData& data, QPowerNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(QPowerData& data, QPowerNode* tree) {
  vec lambda = PredictPois(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double quasi_likelihood(double A, double B, double lambda, double p, double phi) {
  return A * exp(lambda * (1. - p)) / (1. - p) / phi
      - B * exp(lambda * (2. - p)) / (2. - p) / phi;
}

double quasi_fisher(double A, double B, double lambda, double p, double phi) {
  return (2. - p) * B * exp(lambda * (2. - p)) / phi -
    (1. - p) * A * exp(lambda * (1. - p)) / phi;
}

double LogLT(QPowerNode* root, const QPowerData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi(), root->pois_params->get_p());
  std::vector<QPowerNode*> leafs = leaves(root);

  double out = 0.0;
  int num_leaves = leafs.size();
  double p = root->pois_params->get_p();
  double phi = root->pois_params->get_phi();
  double s = root->pois_params->get_scale_lambda();
  double s_sq = pow(s,2);
  double prec = 1./s_sq;

  for(int i = 0; i < num_leaves; i++) {
    double A = leafs[i]->ss.A;
    double B = leafs[i]->ss.B;
    if(A != 0.) {
      double lambda_hat = log(A / B);
      double ell_hat = quasi_likelihood(A, B, lambda_hat, p, phi);
      double fish_hat = quasi_fisher(A, B, lambda_hat, p, phi);
      double fish_inv = 1. / fish_hat;
      out += ell_hat - 0.5 * pow(lambda_hat, 2) / (fish_inv + s_sq)
        + 0.5 * log(fish_inv) - 0.5 * log(s_sq + fish_inv);
    }
  }
  return out;
}

void UpdateParams(QPowerNode* root, const QPowerData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi(), root->pois_params->get_p());
  std::vector<QPowerNode*> leafs = leaves(root);
  int num_leaves = leafs.size();

  double p = root->pois_params->get_p();
  double phi = root->pois_params->get_phi();
  double s = root->pois_params->get_scale_lambda();
  double s_sq = pow(s,2);
  double prec = 1./s_sq;

  for(int i = 0; i < num_leaves; i++) {
    double A = leafs[i]->ss.A;
    double B = leafs[i]->ss.B;
    if(A != 0.) {
      double lambda_hat = log(A / B);
      double fish_hat = quasi_fisher(A, B, lambda_hat, p, phi);
      double fish_inv = 1. / fish_hat;

      double m_up = fish_hat * lambda_hat / (fish_hat + prec);
      double v_up = 1. / (fish_hat + prec);
      double s_up = sqrt(v_up);
      leafs[i]->lambda = m_up + s_up * norm_rand();
    }
    else {
      leafs[i]->lambda = s * norm_rand();
    }
  }
}
