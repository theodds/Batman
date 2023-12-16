#ifndef RVAR_FOREST_H
#define RVAR_FOREST_H

#include <RcppArmadillo.h>
#include "RVarNode.h"
#include "RVarParams.h"
#include "mcmc.h"

struct RVarForest {
  std::vector<RVarNode*> trees;
  TreeHypers* tree_hypers;
  RVarParams* var_params;
  int num_gibbs;

  RVarForest(int num_trees, TreeHypers* tree_hypers_, RVarParams* var_params_)
  {
    num_gibbs = 0;
    trees.resize(0);
    tree_hypers = tree_hypers_;
    var_params = var_params_;
    for(int t = 0; t < num_trees; t++) {
      RVarNode* n = new RVarNode(tree_hypers_, var_params_);
      trees.push_back(n);
    }
  }
  
  RVarForest(Rcpp::List hypers, Rcpp::List opts) {
    num_gibbs = 0;
    arma::sp_mat probs = hypers["probs"];
    tree_hypers = new TreeHypers(probs);
    
    double sigma_scale_log_tau = hypers["sigma_scale_log_tau"];
    double shape_tau_0 = hypers["shape_tau_0"];
    double rate_tau_0 = hypers["rate_tau_0"];
    bool update_scale_log_tau = hypers["update_scale_log_tau"];
    var_params = new RVarParams(sigma_scale_log_tau,
                                sigma_scale_log_tau,
                                shape_tau_0,
                                rate_tau_0,
                                update_scale_log_tau);
    
    int num_trees = hypers["num_tree"];
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      RVarNode* n = new RVarNode(tree_hypers, var_params);
      trees.push_back(n);
    }
  }

  ~RVarForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }

  arma::vec get_s() {return tree_hypers->get_s();}
  arma::uvec get_counts() {return get_var_counts(trees);}
  double get_sigma_mu() {return var_params->get_scale_log_tau();}
  double get_sigma() {return pow(var_params->tau_0, -0.5);}
  arma::vec do_predict(const arma::mat& X);
  arma::mat do_gibbs(const arma::mat& X, const arma::vec& Y,
                     const arma::mat& X_test, int num_iter);

};

arma::vec PredictVar(std::vector<RVarNode*>& forest, const arma::mat& X);
void UpdateHypers(RVarParams& var_params,
                  std::vector<RVarNode*>& trees,
                  const RVarData& data);
void get_params(RVarNode* n, std::vector<double>& tau);
void UpdateTau0(RVarParams& var_params, RVarData& data);
void UpdateScaleLogTau(RVarParams& var_params, std::vector<double>& tau);

#endif
