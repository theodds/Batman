#include "MLogitForest.h"

using namespace arma;
using namespace Rcpp;

arma::mat PredictMLogit(std::vector<MLogitNode*>& forest, const arma::mat& X)
{
  int N = forest.size();
  int num_cat = forest[0]->lambda.size();
  mat out = zeros<mat>(X.n_rows, num_cat);
  for(int n = 0; n < N; n++) {
    out = out + PredictMLogit(forest[n], X);
  }
  return out;
}

void UpdateLambda0(MLogitParams& mlogit_params, MLogitData& data)
{
  int N = data.lambda_hat.n_rows;
  int K = data.lambda_hat.n_cols;

  // Backfit and compute counts
  vec counts = zeros<vec>(K);
  for(int n = 0; n < N; n++) {
    unsigned int y = data.Y(n);
    counts(y) = counts(y) + 1.0;
    for(int k = 0; k < K; k++) {
      data.lambda_hat(n,k)
        = data.lambda_hat(n,k) - mlogit_params.lambda_0(k);
    }
  }

  // Update
  mat exp_lambda_minus = exp(data.lambda_hat);
  for(int k = 0; k < K; k++) {

    double alpha_up = mlogit_params.shape_lambda_0 + counts(k);
    double beta_up  = mlogit_params.rate_lambda_0;

    for(int n = 0; n < N; n++) {
      beta_up += data.phi(n) * exp_lambda_minus(n,k);
    }

    mlogit_params.lambda_0(k) = rlgam(alpha_up) - log(beta_up);
  }


  // Refit
  for(int n = 0; n < N; n++) {
    for(int k = 0; k < K; k++) {
      data.lambda_hat(n,k)
        = data.lambda_hat(n,k) + mlogit_params.lambda_0(k);
    }
  }

}

void UpdatePhi(MLogitData& data, MLogitParams& mlogit_params)
{
  mat exp_lambda_hat = exp(data.lambda_hat);
  vec rate_vec = sum(exp_lambda_hat, 1);
  for(int i = 0; i < data.phi.size(); i++) {
    data.phi(i) = R::rgamma(1.0, 1.0 / rate_vec(i));
  }
}

void UpdateHypers(MLogitParams& mlogit_params,
                  std::vector<MLogitNode*>& trees,
                  MLogitData& data)
{

  vec lambda = get_params(trees);

  UpdateLambda0(mlogit_params,data);
  UpdatePhi(data,mlogit_params);
  UpdateScaleLambda(mlogit_params,lambda);
}

void get_params(MLogitNode* n, std::vector<double>& mu)
{
  if(n->is_leaf) {
    for(int i = 0; i < n->lambda.size(); i++) {
      mu.push_back(n->lambda(i));
    }
  }
  else {
    get_params(n->left, mu);
    get_params(n->right, mu);
  }
}

arma::vec get_params(std::vector<MLogitNode*>& forest) {
  std::vector<double> lambda;
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_params(forest[t], lambda);
  }
  int num_leaves = lambda.size();
  vec lambda_out = zeros<vec>(num_leaves);
  for(int t = 0; t < num_leaves; t++) lambda_out(t) = lambda[t];

  return lambda_out;
}

void UpdateScaleLambda(MLogitParams& mlogit_params,
                       const arma::vec& lambda)
                       // std::vector<MLogitNode*>& trees)
{
  double n = (double)lambda.size();
  double sum_lambda = sum(lambda);
  double sum_exp_lambda = sum(exp(lambda));
  double scale = mlogit_params.sigma_scale_lambda;
  ScaleLambdaLoglik* loglik
    = new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  double scale_0 = mlogit_params.get_scale_lambda();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  mlogit_params.set_scale_lambda(scale_1);

  delete loglik;
}

arma::mat LambdaToPi(const arma::mat& lambda) {
  mat pi = zeros<mat>(lambda.n_rows, lambda.n_cols);
  for(int i = 0; i < lambda.n_rows; i++) {
    vec lambda_vec = trans(lambda.row(i));
    pi.row(i) = trans(exp(lambda_vec - log_sum_exp(lambda_vec)));
  }
  return pi;
}

// [[Rcpp::export]]
List MLogitBart(const arma::mat& X,
                const arma::uvec& Y,
                const arma::sp_mat& probs,
                int num_cat,
                int num_trees,
                double scale_lambda,
                double shape_lambda_0,
                double rate_lambda_0,
                int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  MLogitParams mlogit_params(scale_lambda,
                             shape_lambda_0,
                             rate_lambda_0,
                             num_cat);
  MLogitForest forest(num_trees, &tree_hypers, &mlogit_params);
  MLogitData data(X, Y, num_cat);
  cube pi     = zeros<cube>(Y.size(), num_cat, num_save);
  cube lambda = zeros<cube>(Y.size(), num_cat, num_save);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec scale_lambda_out = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, mlogit_params, tree_hypers);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing iteration " << iter + 1 << "\t\t\t";
    }
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, mlogit_params, tree_hypers);
    }
    lambda.slice(iter) = data.lambda_hat;
    pi.slice(iter) = LambdaToPi(data.lambda_hat);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    scale_lambda_out(iter) = mlogit_params.get_scale_lambda();
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing iteration " << iter + 1 << "\t\t\t";
    }
  }

  List out;
  out["lambda"] = lambda;
  out["pi"]     = pi;
  out["counts"] = counts;
  out["scale_lambda"] = scale_lambda_out;

  return out;
}
