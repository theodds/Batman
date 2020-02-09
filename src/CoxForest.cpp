#include "CoxForest.h"

// TODO: Add update for Phi

using namespace arma;
using namespace Rcpp;

arma::vec PredictCox(std::vector<CoxNode*>& forest, const arma::mat& X) {
  int T = forest.size();
  vec out = zeros<vec>(X.n_rows);
  for(int t = 0; t < T; t++) {
    out = out + PredictCox(forest[t], X);
  }
  return out;
}

void UpdateHypers(CoxParams& cox_params,
                  std::vector<CoxNode*>& trees,
                  CoxData& data)
{

  UpdatePhi(data);

  std::vector<double> lambda;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], lambda);
  }

  UpdateScaleLambda(cox_params, lambda);
}



void get_params(CoxNode* n, std::vector<double>& lambda)
{
  if(n->is_leaf) {
    lambda.push_back(n->lambda);
  }
  else {
    get_params(n->left, lambda);
    get_params(n->right, lambda);
  }
}

void UpdateScaleLambda(CoxParams& cox_params, std::vector<double>& lambda)
{
  double n = (double)lambda.size();
  double sum_lambda = 0.;
  double sum_exp_lambda = 0.;
  for(int i = 0; i < lambda.size(); i++) {
    sum_lambda += lambda[i];
    sum_exp_lambda += exp(lambda[i]);
  }
  double scale = cox_params.sigma_scale_lambda;
  ScaleLambdaLoglik* loglik =
    new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  double scale_0 = cox_params.get_scale_lambda();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  cox_params.set_scale_lambda(scale_1);

  delete loglik;
}

// [[Rcpp::export]]
List CoxBart(const arma::mat& X,
             const arma::vec& Y,
             const arma::uvec& delta,
             const arma::uvec& order,
             const arma::uvec& L,
             const arma::uvec& U,
             const arma::sp_mat& probs,
             const arma::mat& X_test,
             int num_trees,
             double scale_lambda,
             int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  CoxParams cox_params(scale_lambda, scale_lambda);
  CoxForest forest(num_trees, &tree_hypers, &cox_params);
  CoxData data(X, Y, delta, order, L, U);
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  mat lambda_train = zeros<mat>(num_save, X.n_rows);
  mat phi = zeros<mat>(num_save, Y.size());
  umat counts = zeros<umat>(num_save, probs.n_cols);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, cox_params, tree_hypers);
    data.Shuffle();
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing warmup " << iter+1 << "\t\t\t\t";
    }
  }
  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, cox_params, tree_hypers);
      data.Shuffle();
    }
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing save " << iter+1 << "\t\t\t\t";
    }
    lambda_test.row(iter) = trans(PredictCox(forest.trees, X_test));
    lambda_train.row(iter) = trans(data.lambda_hat);
    phi.row(iter) = trans(data.phi);
    counts.row(iter) = trans(get_var_counts(forest.trees));
  }

  List out;
  out["lambda_test"] = lambda_test;
  out["lambda_train"] = lambda_train;
  out["hazard"] = phi;
  out["counts"] = counts;
  out["order"] = data.order;
  return out;
}




































