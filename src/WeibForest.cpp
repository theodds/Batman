#include "WeibForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictWeib(std::vector<WeibNode*>& forest, const arma::mat& X)
{
  int N = forest.size();
  vec out = forest[0]->weib_params->lambda_0 * ones<vec>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictWeib(forest[n], X);
  }
  return out;
}

void UpdateLambda0(WeibParams& weib_params, WeibData& data)
{
  int N = data.mu_hat.n_rows;

  // Backfit
  for(int n = 0; n < N; n++) {
    data.mu_hat(n) = data.mu_hat(n) - weib_params.lambda_0;
  }

  // Compute alpha_up and beta_up
  double alpha_up = weib_params.shape_lambda_0;
  double beta_up = weib_params.rate_lambda_0;
  for(int i = 0; i < data.X.n_rows; i++) {
    alpha_up += data.W[i].size();
    beta_up += exp(data.mu_hat(i)) * pow(data.Y(i), weib_params.weibull_power);
  }

  weib_params.lambda_0 = rlgam(alpha_up) - log(beta_up);

  // Refit
  for(int n = 0; n < N; n++) {
    data.mu_hat(n) = data.mu_hat(n) + weib_params.lambda_0;
  }

}

void get_params(WeibNode* n, std::vector<double>& lambda) {
  if(n->is_leaf) {
    lambda.push_back(n->lambda);
  }
  else {
    get_params(n->left, lambda);
    get_params(n->right, lambda);
  }
}

arma::vec get_params(std::vector<WeibNode*>& forest) {
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

void UpdateHypers(WeibParams& weib_params,
                  std::vector<WeibNode*>& trees,
                  WeibData& data)
{
  vec lambda = get_params(trees);

  UpdateLambda0(weib_params, data);
  if(weib_params.update_scale) UpdateScaleLambda(weib_params, lambda);
}

void UpdateScaleLambda(WeibParams& weib_params, const arma::vec& lambda)
{
  double n = (double)lambda.size();
  double sum_lambda = sum(lambda);
  double sum_exp_lambda = sum(exp(lambda));
  double scale = weib_params.sigma_scale_lambda;
  ScaleLambdaLoglik* loglik
    = new ScaleLambdaLoglik(n,sum_lambda,sum_exp_lambda,scale);
  double scale_0 = weib_params.get_scale_lambda();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  weib_params.set_scale_lambda(scale_1);
}

// [[Rcpp::export]]
List WeibBart(const arma::mat& X,
              const arma::vec& Y,
              const arma::vec& W,
              const arma::uvec& idx,
              const arma::sp_mat& probs,
              int num_trees,
              double scale_lambda,
              double shape_lambda_0,
              double rate_lambda_0,
              double weibull_power,
              bool do_ard,
              bool update_alpha,
              bool update_scale,
              int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  tree_hypers.update_s = do_ard;
  tree_hypers.update_alpha = update_alpha;

  WeibParams weib_params(scale_lambda,
                         shape_lambda_0,
                         rate_lambda_0,
                         weibull_power,
                         update_scale);
  WeibForest forest(num_trees, &tree_hypers, &weib_params);
  WeibData data(X,Y,W,idx);
  mat lambda = zeros<mat>(num_save, Y.size());
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec scale_lambda_out = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, weib_params, tree_hypers);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, weib_params, tree_hypers);
    }
    lambda.row(iter) = data.mu_hat.t();
    counts.row(iter) = trans(get_var_counts(forest.trees));
    scale_lambda_out(iter) = weib_params.get_scale_lambda();
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing save " << iter + 1 << "\t\t\t";
    }
  }

  List out;
  out["lambda"] = lambda;
  out["counts"] = counts;
  out["scale_lambda"] = scale_lambda_out;

  return out;

}

void WeibModel::set_s(const arma::vec& s_) {
  tree_hypers.set_s(s_);
}

arma::vec WeibModel::get_s() {
  return tree_hypers.get_s();
}

arma::vec WeibModel::predict_vec(const arma::mat& X_test) {
  return PredictWeib(forest.trees, X_test);
}

Rcpp::List WeibModel::get_params() {

  List out;
  out["alpha"] = tree_hypers.alpha;
  out["sigma_lambda"] = weib_hypers.get_scale_lambda();
  out["lambda_0"] = weib_hypers.lambda_0;
  out["weibull_power"] = weib_hypers.weibull_power;

  return out;
}

arma::uvec WeibModel::get_counts() {
  return get_var_counts(forest.trees);
}

arma::mat WeibModel::do_gibbs(const arma::mat& X,
                               const arma::vec& Y,
                               const arma::vec& W,
                               const arma::uvec& idx,
                               const arma::mat& X_test,
                               int num_iter) {

  WeibData data(X, Y, W, idx);
  data.mu_hat = PredictWeib(forest.trees, X);
  mat rate_out = zeros<mat>(num_iter, X_test.n_rows);

  for(int i = 0; i < num_iter; i++) {
    IterateGibbs(forest.trees, data, weib_hypers, tree_hypers);
    rate_out.row(i) = trans(PredictWeib(forest.trees, X_test));
    if((i+1) % 100 == 0) {
      Rcout << "Finishing iteration " << i+1 << std::endl;
    }
  }

  return rate_out;

}

WeibModel::WeibModel(arma::sp_mat probs,
                     int num_trees,
                     double scale_lambda,
                     double shape_lambda_0,
                     double rate_lambda_0,
                     double weibull_power,
                     bool update_scale)
  : tree_hypers(probs),
    weib_hypers(scale_lambda, shape_lambda_0, rate_lambda_0,
                weibull_power, update_scale),
    forest(num_trees, &tree_hypers, &weib_hypers)
{
}


// WeibModel::~WeibModel() {
//   for(int i = 0; i < forest.trees.size(); i++) delete forest.trees[i];
// }

void WeibModel::do_ard() {
  tree_hypers.update_s = true;
  tree_hypers.update_alpha = true;
}

RCPP_MODULE(weib_forest) {
  class_<WeibModel>("WeibModel")

    .constructor<arma::sp_mat, int, double, double, double, double, bool>()
    .method("do_gibbs", &WeibModel::do_gibbs)
    .method("get_s", &WeibModel::get_s)
    .method("set_s", &WeibModel::set_s)
    .method("get_params", &WeibModel::get_params)
    .method("predict", &WeibModel::predict_vec)
    .method("get_counts", &WeibModel::get_counts)
    .method("do_ard", &WeibModel::do_ard)

    ;
}

