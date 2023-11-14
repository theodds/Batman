#include "CoxPEForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictCox(std::vector<CoxPENode*>& forest, const arma::mat& X) {
  int T = forest.size();
  vec out = zeros<vec>(X.n_rows);
  for(int t = 0; t < T; t++) {
    out = out + PredictCox(forest[t], X);
  }
  return out;
}

void UpdateHypers(CoxPEParams& cox_params,
                  std::vector<CoxPENode*>& trees,
                  CoxPEData& data)
{

  data.UpdateBase();

  std::vector<double> lambda;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], lambda);
  }

  UpdateScaleLambda(cox_params, lambda);
}



void get_params(CoxPENode* n, std::vector<double>& lambda)
{
  if(n->is_leaf) {
    lambda.push_back(n->lambda);
  }
  else {
    get_params(n->left, lambda);
    get_params(n->right, lambda);
  }
}

void UpdateScaleLambda(CoxPEParams& cox_params, std::vector<double>& lambda)
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
std::vector<std::vector<int>> convertListToVector(Rcpp::List list) {
  std::vector<std::vector<int>> result;
  
  for(int i = 0; i < list.size(); ++i) {
    Rcpp::IntegerVector vec = list[i];
    std::vector<int> stdVec = Rcpp::as<std::vector<int>>(vec);
    result.push_back(stdVec);
  }
  
  return result;
}

// [[Rcpp::export]]
List CoxPEBart(const arma::mat& X,
               const arma::vec& Y,
               const arma::uvec& delta,
               Rcpp::List bin_to_obs_list,
               const arma::uvec& obs_to_bin,
               const arma::vec& time_grid,
               const arma::vec& bin_width,
               const arma::vec& base_haz_init,
               const arma::sp_mat& probs,
               const arma::mat& X_test,
               int num_trees,
               double scale_lambda,
               bool do_rel_surv,
               const arma::vec& pop_haz_,
               int num_burn, int num_thin, int num_save)
{
  std::vector<std::vector<int>> bin_to_obs 
    = convertListToVector(bin_to_obs_list);
  vec pop_haz = do_rel_surv ? pop_haz_ : zeros<vec>(Y.n_elem);
  
  TreeHypers tree_hypers(probs);
  CoxPEParams cox_params(scale_lambda, scale_lambda);
  CoxPEForest forest(num_trees, &tree_hypers, &cox_params);
  CoxPEData data(X, Y, delta, bin_to_obs, obs_to_bin, time_grid,
               bin_width, base_haz_init, pop_haz);
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  mat lambda_train = zeros<mat>(num_save, X.n_rows);
  mat base_haz = zeros<mat>(num_save, bin_width.n_elem);
  mat cum_base_haz = zeros<mat>(num_save, bin_width.n_elem);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec loglik = zeros<vec>(num_save);
  vec sigma_lambda = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, cox_params, tree_hypers);
    if(do_rel_surv) data.RelSurvDA();
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing warmup " << iter+1 << "\t\t\t\t";
    }
  }
  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, cox_params, tree_hypers);
      if(do_rel_surv) data.RelSurvDA();
    }
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing save " << iter+1 << "\t\t\t\t";
    }
    lambda_test.row(iter) = trans(PredictCox(forest.trees, X_test));
    lambda_train.row(iter) = trans(data.lambda_hat);
    base_haz.row(iter) = trans(data.base_haz);
    cum_base_haz.row(iter) = trans(data.cum_base_haz);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    sigma_lambda(iter) = cox_params.get_scale_lambda();

    loglik(iter) = data.loglik;
  }

  List out;
  out["lambda_test"] = lambda_test;
  out["lambda_train"] = lambda_train;
  out["hazard"] = base_haz;
  out["cum_hazard"] = cum_base_haz;
  out["counts"] = counts;
  out["sigma_lambda"] = sigma_lambda;
  out["loglik"] = loglik;
  return out;
}


