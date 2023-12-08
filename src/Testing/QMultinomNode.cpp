#include "QMultinomNode.h"

using namespace arma;
using namespace Rcpp;

void QMultinomNode::AddSuffStat(const QMultinomData& data, int i, double phi) {
  ss.Increment(trans(data.Z.row(i)),
               data.rho(i),
               trans(data.lambda_hat.row(i)),
               phi);
  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data, i, phi);
    } else {
      right->AddSuffStat(data,i, phi);
    }
  }
}

void QMultinomNode::UpdateSuffStat(const QMultinomData& data, double phi) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data,i, phi);
  }
}

arma::vec Predict(QMultinomNode* n, const rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return Predict(n->left, x);
  }
  else {
    return Predict(n->right, x);
  }
}

arma::mat Predict(QMultinomNode* tree, const arma::mat& X) {
  int N = X.n_rows;
  int K = tree->lambda.n_elem;
  mat out = zeros<vec>(N, K);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = trans(Predict(tree, x));
  }
  return out;
}

void BackFit(QMultinomData& data, QMultinomNode* tree) {
  mat lambda = Predict(tree, data.X);
  data.lambda_hat = data.lambda_hat - lambda;
}

void Refit(QMultinomData& data, QMultinomNode* tree) {
  mat lambda = Predict(tree, data.X);
  data.lambda_hat = data.lambda_hat + lambda;
}

double LogLT(QMultinomNode* root, const QMultinomData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QMultinomNode*> leafs = leaves(root);

  double out     = 0.0;
  int num_leaves = leafs.size();
  int K          = root->lambda.n_elem;
  double alpha   = root->pois_params->get_alpha();
  double beta    = root->pois_params->get_beta();

  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < K; k++) {
      double A_ell  = leafs[i]->ss.sum_Z_by_phi(k);
      double B_ell  = leafs[i]->ss.sum_exp_lambda_minus_by_phi(k);
      out += poisson_lgamma(marginal_loglik(A_ell, 0., B_ell, alpha, beta));
    }
  }
  return out;
}

void UpdateParams(QMultinomNode* root, const QMultinomData& data) {
  root->UpdateSuffStat(data, root->pois_params->get_phi());
  std::vector<QMultinomNode*> leafs = leaves(root);
  int num_leaves = leafs.size();
  int K          = root->lambda.n_elem;
  double alpha   = root->pois_params->get_alpha();
  double beta    = root->pois_params->get_beta();
  for(int i = 0; i < num_leaves; i++) {
    for(int k = 0; k < K; k++) {
      double A_ell = leafs[i]->ss.sum_Z_by_phi(k);
      double B_ell = leafs[i]->ss.sum_exp_lambda_minus_by_phi(k);
      leafs[i]->lambda(k)
        = poisson_lgamma_draw_posterior(A_ell, 0., B_ell, alpha, beta);
    }
  }
}
