#include "Batman.h"
using namespace Rcpp;
using namespace arma;

arma::vec PredictMixLoc(const arma::uvec& clust, MixParams& params)
{
  int N = clust.size();
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    out(i) = params.loc_class(clust(i));
  }
  return out;
}

arma::vec PredictMixPrec(const arma::uvec& clust, MixParams& params)
{
  int N = clust.size();
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    out(i) = params.prec_class(clust(i));
  }
  return out;
}

void UpdateMixComp(VarLogitData& data, MixParams& mix_params) {
  // Backfit
  data.var_data.mu_hat = data.var_data.mu_hat - PredictMixLoc(data.mlogit_data.Y, mix_params); 
  data.var_data.tau_hat = data.var_data.tau_hat / PredictMixPrec(data.mlogit_data.Y, mix_params);

  // Update
  data.mlogit_data.Y = update_gmm_class(data.var_data.Y,
                                       data.var_data.mu_hat,
                                       1.0 / sqrt(data.var_data.tau_hat),
                                       mix_params.loc_class,
                                       1.0 / sqrt(mix_params.prec_class),
                                       data.mlogit_data.lambda_hat);

  // Refit
  data.var_data.mu_hat = data.var_data.mu_hat + PredictMixLoc(data.mlogit_data.Y, mix_params); 
  data.var_data.tau_hat = data.var_data.tau_hat % PredictMixPrec(data.mlogit_data.Y, mix_params);
}

void MixUpdateSigmaLoc(MixParams& params) {

  double cauchy_scale = 1.0;
  double new_prec
    = half_cauchy_update_precision_mh(params.loc_class,
                                      pow(params.sigma_loc, -2.0),
                                      cauchy_scale);
  params.sigma_loc = pow(new_prec, -0.5);
  // Rcout << params.sigma_loc << std::endl;
}

void MixUpdateShape(MixParams& params)
{

  params.shape_prec =
    gamma_gamma_shape_update(params.prec_class, 1.0,
                             params.shape_shape, params.rate_shape);
}

void UpdateMix(MixParams& params, VarLogitData& data) {
  int K = params.loc_class.size();
  int N = data.var_data.Y.size();

  // Back fit
  data.var_data.mu_hat = data.var_data.mu_hat - PredictMixLoc(data.mlogit_data.Y, params);

  // Update mu
  vec mu_mu = zeros<vec>(K);
  vec prec_mu = ones<vec>(K) * pow(params.sigma_loc, -2.0);
  for(int i = 0; i < N; i++) {
    int k = data.mlogit_data.Y(i);
    double R = data.var_data.Y(i) - data.var_data.mu_hat(i);
    double tau = data.var_data.tau_hat(i);
    prec_mu(k) = prec_mu(k) + tau;
    mu_mu(k) = mu_mu(k) + tau * R;
  }
  mu_mu = mu_mu / prec_mu;

  for(int k = 0; k < K; k++) {
    params.loc_class(k) =
      mu_mu(k) + pow(prec_mu(k), -0.5) * norm_rand();
  }
  
  // Refit
  data.var_data.mu_hat = data.var_data.mu_hat + PredictMixLoc(data.mlogit_data.Y, params);
  
  // Backfit
  data.var_data.tau_hat = data.var_data.tau_hat / PredictMixPrec(data.mlogit_data.Y, params);
  
  // Update prec
  vec shape_up = ones<vec>(K) * params.shape_prec;
  vec rate_up = ones<vec>(K) * params.shape_prec;
  for(int i = 0; i < N; i++) {
    int k = data.mlogit_data.Y(i);
    double R = data.var_data.Y(i) - data.var_data.mu_hat(i);
    double tau_hat = data.var_data.tau_hat(i); 

    shape_up(k) = shape_up(k) + 0.5;
    rate_up(k) = rate_up(k) + 0.5 * tau_hat * R * R;
  }

  for(int k = 0; k < K; k++) {
    params.prec_class(k) = R::rgamma(shape_up(k), 1.0 / rate_up(k));
  }
  
  // Refit
  data.var_data.tau_hat = data.var_data.tau_hat % PredictMixPrec(data.mlogit_data.Y, params);

  // Update other parameters
  MixUpdateSigmaLoc(params);
  MixUpdateShape(params);
}

arma::mat CalcDensityBatman(const arma::mat& X,
                            const arma::vec& Y,
                            std::vector<VarLogitNode*>& trees,
                            VarLogitParams& params,
                            MixParams& mix_params)
{

  int n_pred = X.n_rows;
  int n_grid = Y.size();
  int n_clust = mix_params.loc_class.size();

  mat out         = zeros<mat>(X.n_rows, Y.size());
  mat mu_tau      = PredictVar(trees, X);
  mu_tau.col(1)   = mu_tau.col(1) * params.var_params.tau_0;
  mat log_weights = PredictMLogit(trees, X);

  for(int i = 0; i < n_pred; i++) {
    vec logw = trans(log_weights.row(i)) + params.mlogit_params.lambda_0;
    logw     = logw - log_sum_exp(logw);
    for(int j = 0; j < n_grid; j++) {
      vec logliks = logw;
      for(int k = 0; k < n_clust; k++) {
        double mu_k = mu_tau(i,0) + mix_params.loc_class(k);
        double sigma_k = 1.0 / sqrt(mu_tau(i,1) * mix_params.prec_class(k));
        logliks(k) = logliks(k) + R::dnorm(Y(j), mu_k, sigma_k, 1);
      }
      out(i,j) = log_sum_exp(logliks);
    }
  }

  return out;
}

// [[Rcpp::export]]
Rcpp::List Batman(const arma::mat& X,
                  const arma::vec& Y,
                  const arma::sp_mat& probs,
                  int num_cat,
                  int num_tree,
                  double scale_lambda,
                  double shape_lambda_0,
                  double rate_lambda_0,
                  double scale_kappa,
                  double sigma_scale_log_tau,
                  double shape_tau_0,
                  double rate_tau_0,
                  int num_burn, int num_thin, int num_save,
                  const arma::mat& X_test,
                  const arma::vec& Y_test)
{

  // Initialize latent variables
  uvec latent_class = zeros<uvec>(Y.size());
  for(int i = 0; i < Y.size(); i++) {
    latent_class(i) = sample_class(num_cat);
  }

  TreeHypers tree_hypers(probs);
  double kappa_init = pow(scale_kappa, -2.0);
  VarLogitParams params(kappa_init, sigma_scale_log_tau, sigma_scale_log_tau,
                        shape_tau_0, rate_tau_0, scale_kappa,
                        scale_lambda, shape_lambda_0, rate_lambda_0, num_cat);
  VarLogitForest forest(num_tree, &tree_hypers, &params, num_cat);
  VarLogitData data(X, latent_class, num_cat, X, Y);
  MixParams mix_params(num_cat);
  // Make sure VarLogitParams is initialized properly
  data.var_data.mu_hat = PredictMixLoc(data.mlogit_data.Y, mix_params);
  data.var_data.tau_hat = PredictMixPrec(data.mlogit_data.Y, mix_params);

  mat mu = zeros<mat>(num_save, Y.size());
  umat class_out = zeros<umat>(num_save, Y.size());
  cube dens_test = zeros<cube>(X_test.n_rows, Y_test.size(), num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, params, tree_hypers);
    UpdateMix(mix_params, data);
    // if(iter > 1000) {
      UpdateMixComp(data, mix_params);
    // }
      
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }
  
  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, params, tree_hypers);
      UpdateMix(mix_params, data);
      UpdateMixComp(data, mix_params);
    }
    
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing save " << iter + 1 << "\t\t\t";
    }
    class_out.row(iter) = trans(data.mlogit_data.Y);
    dens_test.slice(iter) =
      CalcDensityBatman(X_test, Y_test, forest.trees, params, mix_params);
  }

  Rcpp::List out; 
  
  out["mu_hat"] = data.var_data.mu_hat;
  out["tau_hat"] = data.var_data.tau_hat;
  out["shape"] = mix_params.shape_prec;
  out["class"] = class_out;
  out["dens"] = dens_test;
  return out;
  
}


