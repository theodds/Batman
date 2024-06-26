#include "RVarForest.h"

using namespace arma;
using namespace Rcpp;

arma::vec PredictVar(std::vector<RVarNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec tau = ones<vec>(X.n_rows);
  for(int n = 0; n < N; n++) {
    tau = tau % PredictVar(forest[n], X);
  }
  return tau;
}

void get_params(RVarNode* n, std::vector<double>& tau) {
  if(n->is_leaf) {
    tau.push_back(n->tau);
  }
  else {
    get_params(n->left, tau);
    get_params(n->right, tau);
  }
}

arma::vec RVarForest::do_predict(const arma::mat& X) {
  return PredictVar(trees, X) * var_params->tau_0;
}

void UpdateHypers(RVarParams& var_params,
                  std::vector<RVarNode*>& trees,
                  RVarData& data)
{
  std::vector<double> tau;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], tau);
  }
  // Rcout << "Update Tau";
  UpdateTau0(var_params, data);
  // Rcout << "Update Scale";
  UpdateScaleLogTau(var_params, tau);
  // Rcout << "Done";
}

void UpdateTau0(RVarParams& var_params, RVarData& data) {
  int N = data.Y.size();
  double shape_up = var_params.shape_tau_0 + 0.5 * N;
  double rate_up = var_params.rate_tau_0;
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) / var_params.tau_0;
    rate_up += 0.5 * data.tau_hat(i) * pow(data.Y(i), 2.0);
  }
  double scale_up = 1.0 / rate_up;
  var_params.tau_0 = R::rgamma(shape_up, scale_up);
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) * var_params.tau_0;
  }
}

void UpdateScaleLogTau(RVarParams& var_params, std::vector<double>& tau) {
  // If we don't want to update, we don't update
  if(!var_params.update_scale_log_tau) return;

  // Otherwise, we do update using slice sampling
  double n = tau.size();
  double sum_lambda = 0.;
  double sum_exp_lambda = 0.;
  for(int i = 0; i < tau.size(); i++) {
    sum_lambda += log(tau[i]);
    sum_exp_lambda += tau[i];
  }
  double scale = var_params.sigma_scale_log_tau;
  
  double alpha, beta;

  for(int k = 0; k < 30; k++) {
    // Compute old likleihood
    double scale_old = var_params.get_scale_log_tau();
    scale_lambda_to_alpha_beta(alpha, beta, scale_old);
    double loglik_old = n * alpha * log(beta) - n * R::lgammafn(alpha)
      + alpha * sum_lambda - beta * sum_exp_lambda
      - 0.5 * pow(scale_old / scale, 2.0) + log(scale_old);

    // Proposal
    double scale_new = exp(log(scale_old) + 2. * unif_rand() - 1.0);
    scale_lambda_to_alpha_beta(alpha, beta, scale_new);
    double loglik_new= n * alpha * log(beta) - n * R::lgammafn(alpha)
      + alpha * sum_lambda - beta * sum_exp_lambda
      - 0.5 * pow(scale_new / scale, 2.0) + log(scale_new);
  
    if(log(unif_rand()) < loglik_new - loglik_old) {
      var_params.set_scale_log_tau(scale_new);
    }
  }

  // // Compute old likelihood
  // double alpha, beta;
  // double scale_old = var_params.get_scale_log_tau();
  // scale_lambda_to_alpha_beta(alpha, beta, scale_old);
  // double loglik_old = n * alpha * log(beta) 
  //   - n * R::lgammafn(alpha)
  //   + alpha * sum_lambda
  //   - beta * sum_exp_lambda;
  
  // // Sample from the prior
  // double scale_new = fabs(norm_rand() / norm_rand() * scale);
  // scale_lambda_to_alpha_beta(alpha, beta, scale_new);
  // double loglik_new = n * alpha * log(beta) 
  //   - n * R::lgammafn(alpha)
  //   + alpha * sum_lambda
  //   - beta * sum_exp_lambda;

  // if(log(unif_rand()) < loglik_new - loglik_old) {
  //   var_params.set_scale_log_tau(scale_new);
  // }
  
  // ScaleLambdaLoglik* loglik =
  //   new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  // double scale_0 = var_params.get_scale_log_tau();
  // double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  // var_params.set_scale_log_tau(scale_1);

  // delete loglik;
}

// [[Rcpp::export]]
List RVarBart(const arma::mat& X,
              const arma::vec& Y,
              const arma::sp_mat& probs,
              double sigma_scale_log_tau,
              double shape_tau_0, double rate_tau_0,
              int num_trees,
              int num_burn, int num_thin, int num_save,
              bool update_scale_log_tau,
              bool update_s)
{
  TreeHypers tree_hypers(probs);
  tree_hypers.update_s = update_s;
  RVarParams var_params(sigma_scale_log_tau,
                        sigma_scale_log_tau,
                        shape_tau_0,
                        rate_tau_0,
                        update_scale_log_tau);
  RVarForest forest(num_trees, &tree_hypers, &var_params);
  RVarData data(X, Y);
  mat tau = zeros<mat>(num_save, Y.size());
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec scale_lambda = zeros<vec>(num_save);

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, var_params, tree_hypers);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, var_params, tree_hypers);
    }
    tau.row(iter) = trans(data.tau_hat);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    scale_lambda(iter) = var_params.get_scale_log_tau();
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }
  List out;
  out["tau"] = tau;
  out["scale_lambda"] = scale_lambda;
  out["counts"] = counts;
  
  return out;
}

/*
 * Interface for accessing the model from within R
 */

arma::mat RVarForest::do_gibbs(const arma::mat& X, const arma::vec& Y,
                               const arma::mat& X_test, int num_iter) {

  mat tau_out = zeros<mat>(num_iter, X_test.n_rows);

  RVarData data(X, Y);
  data.tau_hat = do_predict(X);

  for(int i = 0; i < num_iter; i++) {
    IterateGibbs(trees, data, *var_params, *tree_hypers);
    tau_out.row(i) = trans(do_predict(X_test));
    num_gibbs++;
    if(num_gibbs % 100 == 99) {
      Rcpp::Rcout << "\rFinishing iteration  " << num_gibbs + 1 << "\t\t\t";
    }
  }
  return tau_out;
}

RCPP_MODULE(var_forest) {

  class_<RVarForest>("RVarForest")

    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &RVarForest::do_gibbs)
    .method("get_s", &RVarForest::get_s)
    .method("get_counts", &RVarForest::get_counts)
    .method("get_sigma_mu", &RVarForest::get_sigma_mu)
    // .method("set_s", &Forest::set_s)
    .method("get_sigma", &RVarForest::get_sigma)
    // .method("set_sigma", &Forest::set_sigma)
    .method("do_predict", &RVarForest::do_predict)
    // .method("get_tree_counts", &Forest::get_tree_counts)
    // .method("predict_iteration", &Forest::predict_iteration)
    ;
  
}
