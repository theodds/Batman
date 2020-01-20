#include "VarForest.h"

using namespace arma;
using namespace Rcpp;


arma::mat PredictVar(std::vector<VarNode*>& forest, const arma::mat& X) {
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

void UpdateHypers(VarParams& var_params,
                  std::vector<VarNode*>& trees,
                  VarData& data)
{
  std::vector<double> mu;
  std::vector<double> tau;
  for(int i = 0; i < trees.size(); i++) {
    get_params(trees[i], mu, tau);
  }
  // Rcout << "Update Tau";
  UpdateTau0(var_params, data);
  // Rcout << "Update Kappa";
  UpdateKappa(var_params, mu, tau);
  // Rcout << "Update Scale";
  UpdateScaleLogTau(var_params, tau);
  // Rcout << "Done";
}


void get_params(VarNode* n, std::vector<double>& mu, std::vector<double>& tau) {
  if(n->is_leaf) {
    mu.push_back(n->mu);
    tau.push_back(n->tau);
  }
  else {
    get_params(n->left, mu, tau);
    get_params(n->right, mu, tau);
  }
}

void UpdateTau0(VarParams& var_params, VarData& data) {
  int N = data.Y.size();
  double shape_up = var_params.shape_tau_0 + 0.5 * N;
  double rate_up = var_params.rate_tau_0;
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) / var_params.tau_0;
    rate_up += 0.5 * data.tau_hat(i) * pow(data.Y(i) - data.mu_hat(i), 2.0);
  }
  double scale_up = 1.0 / rate_up;
  var_params.tau_0 = R::rgamma(shape_up, scale_up);
  for(int i = 0; i < N; i++) {
    data.tau_hat(i) = data.tau_hat(i) * var_params.tau_0;
  }
}

void UpdateKappa(VarParams& var_params,
                 std::vector<double>& mu,
                 std::vector<double>& tau) {
  int num_leaves = mu.size();
  double shape_up = 0.5 * num_leaves + 1;
  double rate_up = 0.;
  for(int l = 0; l < num_leaves; l++) {
    rate_up += 0.5 * tau[l] * pow(mu[l], 2.0);
  }
  double kappa_prop = R::rgamma(shape_up, 1.0 / rate_up);
  double lograt = cauchy_jacobian(kappa_prop, var_params.scale_kappa) -
    cauchy_jacobian(var_params.kappa, var_params.scale_kappa);
  if(log(unif_rand()) < lograt) {
    var_params.kappa = kappa_prop;
  }
}

void UpdateScaleLogTau(VarParams& var_params, std::vector<double>& tau) {
  double n = (double)tau.size();
  double sum_lambda = 0.;
  double sum_exp_lambda = 0.;
  for(int i = 0; i < tau.size(); i++) {
    sum_lambda += log(tau[i]);
    sum_exp_lambda += tau[i];
  }
  double scale = var_params.sigma_scale_log_tau;
  ScaleLambdaLoglik* loglik =
    new ScaleLambdaLoglik(n, sum_lambda, sum_exp_lambda, scale);
  double scale_0 = var_params.get_scale_log_tau();
  double scale_1 = slice_sampler(scale_0, loglik, 1., 0., R_PosInf);
  var_params.set_scale_log_tau(scale_1);

  delete loglik;
}

// [[Rcpp::export]]
List VarBart(const arma::mat& X,
             const arma::vec& Y,
             const arma::sp_mat& probs,
             double scale_kappa,
             double sigma_scale_log_tau,
             double shape_tau_0, double rate_tau_0,
             int num_trees,
             int num_burn, int num_thin, int num_save)
{
  TreeHypers tree_hypers(probs);
  double kappa_init = pow(scale_kappa, -2.0);
  VarParams var_params(kappa_init,
                       sigma_scale_log_tau,
                       sigma_scale_log_tau,
                       shape_tau_0, rate_tau_0,
                       scale_kappa);
  VarForest forest(num_trees, &tree_hypers, &var_params);
  VarData data(X, Y);
  mat mu = zeros<mat>(num_save, Y.size());
  mat tau = zeros<mat>(num_save, Y.size());
  umat counts = zeros<umat>(num_save, probs.n_cols);

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
    mu.row(iter) = trans(data.mu_hat);
    tau.row(iter) = trans(data.tau_hat);
    counts.row(iter) = trans(get_var_counts(forest.trees));
    if(iter % 100 == 99) {
      Rcpp::Rcout << "\rFinishing warmup " << iter + 1 << "\t\t\t";
    }
  }
  List out;
  out["mu"] = mu;
  out["tau"] = tau;
  out["counts"] = counts;

  return out;
}


