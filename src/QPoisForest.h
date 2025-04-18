#ifndef QPOIS_FOREST_H
#define QPOIS_FOREST_H

#include <RcppArmadillo.h>
#include "QPoisNode.h"
#include "QPoisData.h"
#include "mcmc.h"


struct QPoisForest {
  std::vector<QPoisNode*> trees;
  TreeHypers* tree_hypers;
  QPoisParams* params;
  int num_gibbs;

  QPoisForest(int num_trees, TreeHypers* tree_hypers, QPoisParams* pois_params) {
    num_gibbs = 0;
    this->tree_hypers = tree_hypers;
    this->params = pois_params;
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QPoisNode* n = new QPoisNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QPoisForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

  QPoisForest(Rcpp::List hypers, Rcpp::List opts) {
    num_gibbs = 0;
    arma::sp_mat probs = hypers["probs"];
    tree_hypers = new TreeHypers(probs);
    double scale_lambda = hypers["sigma_scale_lambda"];
    double sigma_scale_lambda = hypers["sigma_scale_lambda"];
    bool update_s = hypers["update_s"];
    double phi = hypers["phi"];
    tree_hypers->update_s = update_s;
    params = new QPoisParams(scale_lambda, scale_lambda, phi);
    int num_trees = hypers["num_tree"];

    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QPoisNode* n = new QPoisNode(tree_hypers, params);
      trees.push_back(n);
    }
  }


  arma::vec get_s() {return tree_hypers->get_s();}
  arma::uvec get_counts() {return get_var_counts(trees);}
  double get_sigma_mu() {return params->get_scale_lambda();}
  double get_phi() {return params->get_phi();}
  void set_phi(double phi) {params->phi = phi;}
  arma::vec do_predict(const arma::mat& X);
  
  Rcpp::List  do_gibbs(const arma::mat& X,
                       const arma::vec& Y,
                       const arma::vec& offset,
                       const arma::mat& X_test,
                       int num_iter);

};

arma::vec PredictPois(std::vector<QPoisNode*>& forest, const arma::mat& X);

void UpdateHypers(QPoisParams& hypers, std::vector<QPoisNode*>& trees,
                  const QPoisData& data);

#endif
