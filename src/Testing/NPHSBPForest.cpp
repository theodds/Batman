#include "CoxNPHOForest.h"

using namespace arma;
using namespace Rcpp;

arma::mat PredictCox(std::vector<CoxNPHONode*>& forest, const arma::mat& X, int K) {
  int T = forest.size();
  mat out = zeros<mat>(X.n_rows, K-1);
  for(int t = 0; t < T; t++) {
    out = out + PredictCox(forest[t], X, K);
  }
  return out;
}

void UpdateHypers(CoxNPHOParams& cox_params,
                  std::vector<CoxNPHONode*>& trees,
                  CoxNPHOData& data)
{

  // Update Z
  int N = data.Y.n_elem;
  int K = cox_params.gamma.n_elem + 1;
  for(int i = 0; i < N; i++) {
    if(data.Y(i) < K - 1) {
      double U = R::unif_rand();
      double lambda = exp(cox_params.gamma(data.Y(i)) + data.lambda_hat(i, data.Y(i)));
      data.Z(i,data.Y(i)) = -log(1 - U * (1 - exp(-lambda))) / lambda;
    }
  }
  
  // Update gamma
  vec A = zeros<vec>(K - 1);
  vec B = zeros<vec>(K - 1);
  for(int i = 0; i < N; i++) {
    if(data.Y(i) < K - 1) {
      A(data.Y(i)) += 1; 
      B(data.Y(i)) += data.Z(i,data.Y(i)) * exp(data.lambda_hat(i, data.Y(i)));
    }
    for(int k = 0; k < data.Y(i); k++) {
      B(k) += data.Z(i,k) * exp(data.lambda_hat(i, k));
    }
  }
  
  for(int k = 0; k < K - 1; k++) {
    cox_params.gamma(k) = rlgam(A(k) + cox_params.shape_gamma) - 
      log(B(k) + cox_params.rate_gamma);
  }
  // cox_params.gamma(0) = 0.;

  std::vector<double> lambda;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], lambda);
  }

  // UpdateScaleLambda(cox_params, lambda);
}



void get_params(CoxNPHONode* n, std::vector<double>& lambda)
{
  if(n->is_leaf) {
    lambda.push_back(n->lambda);
  }
  else {
    get_params(n->left, lambda);
    get_params(n->right, lambda);
  }
}

void UpdateScaleLambda(CoxNPHOParams& cox_params, std::vector<double>& lambda)
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
List CoxNPHOBart(const arma::mat& X,
               const arma::vec& Y,
               Rcpp::List bin_to_obs_list,
               const arma::sp_mat& probs,
               const arma::mat& X_test,
               int num_trees,
               double scale_lambda,
               double shape_gamma,
               double rate_gamma,
               int num_burn, int num_thin, int num_save)
{
  std::vector<std::vector<int>> bin_to_obs
    = convertListToVector(bin_to_obs_list);
  int K = bin_to_obs.size();

  TreeHypers tree_hypers(probs);
  CoxNPHOParams cox_params(scale_lambda, scale_lambda, shape_gamma, rate_gamma, K);
  CoxNPHOForest forest(num_trees, &tree_hypers, &cox_params);
  CoxNPHOData data(X, Y, bin_to_obs);
  cube lambda_test = zeros<cube>(X_test.n_rows, K-1, num_save);
  cube lambda_train = zeros<cube>(X.n_rows, K-1, num_save);
  mat gamma = zeros<mat>(num_save, K - 1);
  umat counts = zeros<umat>(num_save, probs.n_cols);
//   vec loglik = zeros<vec>(num_save);
  vec sigma_lambda = zeros<vec>(num_save);
  vec shape_gamma_out = zeros<vec>(num_save);
  vec rate_gamma_out = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, cox_params, tree_hypers);
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing warmup " << iter+1 << "\t\t\t\t";
    }
  }
  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, cox_params, tree_hypers);
    }
    if((iter+1) % 100 == 0) {
      Rcout << "\rFinishing save " << iter+1 << "\t\t\t\t";
    }
    lambda_test.slice(iter) = PredictCox(forest.trees, X_test, K);
    lambda_train.slice(iter) = data.lambda_hat;
    gamma.row(iter) = trans(cox_params.gamma);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    sigma_lambda(iter) = cox_params.get_scale_lambda();
    shape_gamma_out(iter) = cox_params.shape_gamma;
    rate_gamma_out(iter) = cox_params.rate_gamma;

// //     loglik(iter) = data.loglik;
  }

  List out;
  out["lambda_test"] = lambda_test;
  out["lambda_train"] = lambda_train;
  out["gamma"] = gamma;
  out["counts"] = counts;
  out["sigma_lambda"] = sigma_lambda;
//   // out["loglik"] = loglik;
  out["shape_gamma"] = shape_gamma_out;
  out["rate_gamma"] = rate_gamma_out;
  return out;
}


