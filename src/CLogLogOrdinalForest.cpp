#include "CLogLogOrdinalForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictPois(std::vector<CLogLogOrdinalNode*>& forest,
                      const arma::mat& X) {
  int N = forest.size();
  vec out = zeros<mat>(X.n_rows);
  for(int n = 0; n < N; n++) {
    out = out + PredictPois(forest[n], X);
  }
  return out;
}

double sample_texp(double lambda) {
  
  double a = 1. - unif_rand() * (1 - exp(-lambda));
  return -log(a) / lambda;
  
}

void UpdateScaleLambda(CLogLogOrdinalParams& params, std::vector<double>& lambda)
{
  double n = (double)lambda.size();
  double sum_lambda = 0.;
  double sum_exp_lambda = 0.;
  for(int i = 0; i < lambda.size(); i++) {
    sum_lambda += lambda[i];
    sum_exp_lambda += exp(lambda[i]);
  }
  double scale = params.sigma_scale_lambda;
  ScaleLambdaLoglik* loglik =
    new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  double scale_0 = params.get_scale_lambda();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  params.set_scale_lambda(scale_1);
  
  delete loglik;
}

void get_params(CLogLogOrdinalNode* n, std::vector<double>& lambda)
{
  if(n->is_leaf) {
    lambda.push_back(n->lambda);
  }
  else {
    get_params(n->left, lambda);
    get_params(n->right, lambda);
  }
}

void UpdateHypers(CLogLogOrdinalParams& hypers,
                  std::vector<CLogLogOrdinalNode*>& trees,
                  CLogLogOrdinalData& data) {
  
  int N = data.Y.n_elem;
  int K = hypers.gamma.n_elem + 1;
  
  // Update Scale
  std::vector<double> lambda;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], lambda);
  }
  
  // UpdateScaleLambda(hypers, lambda);

  // Update the truncated exponentials
  for(int i = 0; i < N; i++) {
    if(data.Y(i) == K - 1) {
      data.Z(i) = 1.;
    }
    else {
      double exp_lambda = exp(data.lambda_hat(i) + hypers.gamma(data.Y(i)));
      data.Z(i) = sample_texp(exp_lambda);
    }
  }

  // Update the gammas

  // arma::vec shapes = arma::ones<arma::vec>(K - 1) * hypers.alpha_gamma;
  // arma::vec rates = arma::ones<arma::vec>(K - 1) * hypers.beta_gamma;
  
  // for(int i = 0; i < N; i++) {
  //   double exp_lambda = exp(data.lambda_hat(i));
  //   for(int k = 0; k < hypers.gamma.n_elem; k++) {
  //     if(data.Y(i) == k) {
  //       shapes(k) += 1;
  //       rates(k) += data.Z(i) * exp_lambda;
  //     }
  //     if(data.Y(i) > k) {
  //       rates(k) += exp_lambda;
  //     }
  //   }
  // }
  vec A = zeros<vec>(K - 1);
  vec B = zeros<vec>(K - 1);
  for(int i = 0; i < N; i++) {
    if(data.Y(i) < K - 1) {
      A(data.Y(i)) += 1;
      B(data.Y(i)) += data.Z(i) * exp(data.lambda_hat(i));
    }
    for(int k = 0; k < data.Y(i); k++) {
      B(k) += 1. * exp(data.lambda_hat(i));
    }
  }

  for(int k = 0; k < hypers.gamma.n_elem; k++) {
    // hypers.gamma(k) = rlgam(shapes(k)) - log(rates(k));
    hypers.gamma(k) = rlgam(A(k) + hypers.alpha_gamma) - log(B(k) + hypers.beta_gamma);
  }
  hypers.gamma(0) = hypers.gamma_0;
  
  // Update the segs
  for(int j = 0; j < hypers.seg.n_elem; j++) {
    if(j == 0) {
      hypers.seg(j) = 0;
    } else {
      hypers.seg(j) = hypers.seg(j - 1) + exp(hypers.gamma(j-1));
    }
  }
  
}

// [[Rcpp::export]]
List CLogLogOrdinalBart(const arma::mat& X,
                        const arma::uvec& Y,
                        int num_levels,
                        const arma::mat& X_test,
                        const arma::sp_mat& probs,
                        int num_trees,
                        double scale_lambda,
                        double alpha_gamma,
                        double beta_gamma,
                        double gamma_0,
                        int num_burn,
                        int num_thin,
                        int num_save
                        )
{

  TreeHypers tree_hypers(probs);
  CLogLogOrdinalParams params(scale_lambda, alpha_gamma, beta_gamma, gamma_0,
                              num_levels);

  Rcout << "CLogLogOrdinalForest forest(num_trees, &tree_hypers, &params);" << std::endl;
  CLogLogOrdinalForest forest(num_trees, &tree_hypers, &params);
  Rcout << "CLogLogOrdinalData data(X, Y);" << std::endl;
  CLogLogOrdinalData data(X, Y);
  mat lambda = zeros<mat>(num_save, Y.size());
  mat lambda_test = zeros<mat>(num_save, X_test.n_rows);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  mat gamma = zeros<mat>(num_save, num_levels - 1);
  vec sigma_mu = zeros<vec>(num_save);

  Rcout << "for(int iter = 0; iter < num_burn; iter++) {" << std::endl;
  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, params, tree_hypers);
    if(iter % 100 == 0) Rcout << "\rFinishing warmup iteration "
                              << iter << "\t\t\t";
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, params, tree_hypers);
    }
    if(iter % 100 == 0) Rcout << "\rFinishing save iteration "
                              << iter << "\t\t\t";
    lambda.row(iter) = trans(data.lambda_hat);
    lambda_test.row(iter) = trans(PredictPois(forest.trees, X_test));
    counts.row(iter) = trans(get_var_counts(forest.trees));
    gamma.row(iter) = trans(params.gamma);
    sigma_mu(iter) = params.get_scale_lambda();
  }

  List out;
  out["lambda"] = lambda;
  out["lambda_test"] = lambda_test;
  out["counts"] = counts;
  out["gamma"] = gamma;
  out["sigma_mu"] = sigma_mu;

  return out;
  
}


