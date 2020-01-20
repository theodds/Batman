#include "VarLogitForest.h"

using namespace arma;
using namespace Rcpp;

arma::mat PredictMLogit(std::vector<VarLogitNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  int num_cat = forest[0]->lambda.size();
  mat out = zeros<mat>(X.n_rows, num_cat);
  for(int n = 0; n < N; n++) {
    out = out + PredictMLogit(forest[n], X);
  }
  return out;
}

arma::mat PredictVar(std::vector<VarLogitNode*>& forest, const arma::mat& X) {
  int N = forest.size();
  vec mu = zeros<vec>(X.n_rows);
  vec tau = ones<vec>(X.n_rows);
  for(int n = 0; n < N; n++) {
    mat tmp = PredictVar(forest[n], X);
    mu = mu + tmp.col(0);
    tau = tau % tmp.col(1);
  }
  return join_rows(mu,tau);
}

void UpdatePhi(VarLogitData& data, VarLogitParams& mlogit_params)
{
  mat exp_lambda_hat = exp(data.mlogit_data.lambda_hat);
  vec rate_vec = sum(exp_lambda_hat, 1);
  for(int i = 0; i < data.mlogit_data.phi.size(); i++) {
    data.mlogit_data.phi(i) = R::rgamma(1.0, 1.0 / rate_vec(i));
  }
}

void UpdateHypers(VarLogitParams& params,
                  std::vector<VarLogitNode*>& trees,
                  VarLogitData& data) {

  // Logit params
  std::vector<double> lambda_;
  get_lambda(trees, lambda_);
  vec lambda = zeros<vec>(lambda_.size());
  for(int i = 0; i < lambda.size(); i++) lambda(i) = lambda_[i];


  UpdatePhi(data, params);
  UpdateLambda0(params.mlogit_params, data.mlogit_data);
  UpdateScaleLambda(params.mlogit_params, lambda);


  // Var params
  std::vector<double> mu;
  std::vector<double> tau;
  get_mu_tau(trees, mu, tau);

  UpdateTau0(params.var_params, data.var_data);
  UpdateKappa(params.var_params, mu, tau);
  UpdateScaleLogTau(params.var_params, tau);


}

void get_mu_tau(std::vector<VarLogitNode*> trees, std::vector<double>& mu, std::vector<double>& tau) {
  int num_trees = trees.size();
  for(int l = 0; l < num_trees; l++) {
    std::vector<VarLogitNode*> leafs = leaves(trees[l]);
    for(int k = 0; k < leafs.size(); k++) {
      mu.push_back(leafs[k]->mu);
      tau.push_back(leafs[k]->tau);
    }
  }
}

void get_lambda(std::vector<VarLogitNode*> trees, std::vector<double>& lambda) {
  int num_trees = trees.size();
  for(int l = 0; l < num_trees; l++) {
    std::vector<VarLogitNode*> leafs = leaves(trees[l]);
    for(int k = 0; k < leafs.size(); k++) {
      for(int n = 0; n < leafs[k]->lambda.size(); n++) {
        lambda.push_back(leafs[k]->lambda(n));
      }
    }
  }
}


// [[Rcpp::export]]
List VarLogitBart(const arma::mat& X_logit,
                  const arma::uvec& Y_logit,
                  const arma::mat& X_var,
                  const arma::vec& Y_var,
                  const arma::sp_mat& probs,
                  int num_cat,
                  int num_trees,
                  double scale_lambda,
                  double shape_lambda_0,
                  double rate_lambda_0,
                  double scale_kappa, double sigma_scale_log_tau,
                  double shape_tau_0, double rate_tau_0,
                  int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  double kappa_init = pow(scale_kappa, -2.0);
  VarLogitParams params(kappa_init,
                        sigma_scale_log_tau,
                        sigma_scale_log_tau,
                        shape_tau_0,
                        rate_tau_0,
                        scale_kappa,
                        scale_lambda,
                        shape_lambda_0,
                        rate_lambda_0,
                        num_cat);
  VarLogitForest forest(num_trees, &tree_hypers, &params, num_cat);
  VarLogitData data(X_logit, Y_logit, num_cat, X_var, Y_var);


  cube pi     = zeros<cube>(Y_logit.size(), num_cat, num_save);
  cube lambda = zeros<cube>(Y_logit.size(), num_cat, num_save);
  umat counts = zeros<umat>(num_save, probs.n_cols);
  vec scale_lambda_out = zeros<vec>(num_save);
  mat mu = zeros<mat>(num_save, Y_var.size());
  mat tau = zeros<mat>(num_save, Y_var.size());

  for(int iter = 0; iter < num_burn; iter++) {
    IterateGibbs(forest.trees, data, params, tree_hypers);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }

  for(int iter = 0; iter < num_save; iter++) {
    for(int j = 0; j < num_thin; j++) {
      IterateGibbs(forest.trees, data, params, tree_hypers);
    }
    lambda.slice(iter) = data.mlogit_data.lambda_hat;
    pi.slice(iter) = LambdaToPi(data.mlogit_data.lambda_hat);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    scale_lambda_out(iter) = params.mlogit_params.get_scale_lambda();
    mu.row(iter) = trans(data.var_data.mu_hat);
    tau.row(iter) = trans(data.var_data.tau_hat);
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing save " << iter + 1 << "\t\t\t";
    }
  }

  List out;
  out["lambda"] = lambda;
  out["pi"]     = pi;
  out["counts"] = counts;
  out["mu"] = mu;
  out["tau"] = tau;
  out["scale_lambda"] = scale_lambda_out;

  return out;
}
