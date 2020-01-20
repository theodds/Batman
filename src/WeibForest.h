#ifndef WEIB_FOREST_H
#define WEIB_FOREST_H

#include <RcppArmadillo.h>
#include "WeibNode.h"
#include "WeibParams.h"
#include "mcmc.h"

struct WeibForest {
  std::vector<WeibNode*> trees;

  WeibForest(int num_trees, TreeHypers* tree_hypers_, WeibParams* weib_params_)
  {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      WeibNode* n = new WeibNode(tree_hypers_, weib_params_);
      trees.push_back(n);
    }
  }

  ~WeibForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

class WeibModel {

 private:

  TreeHypers tree_hypers;
  WeibParams weib_hypers;
  WeibForest forest;

 public: 

   WeibModel(arma::sp_mat probs,
             int num_trees,
             double scale_lambda,
             double shape_lambda_0,
             double rate_lambda_0,
             double weibull_power);
  ~WeibModel();

  arma::mat do_gibbs(const arma::mat& X,
           const arma::vec& Y,
           const arma::vec& W,
           const arma::uvec& idx,
           const arma::mat& X_test,
           int num_iter);
  void do_ard();

  void set_s(const arma::vec& s);
  arma::vec get_s();
  arma::uvec get_counts();
  Rcpp::List get_params();
  
  // Prediction interface
  arma::vec predict_vec(const arma::mat& X_test);

};

arma::vec PredictWeib(std::vector<WeibNode*>& forest, const arma::mat& X);
void UpdateLambda0(WeibParams& weib_params, WeibData& data);
void get_params(WeibNode* n, std::vector<double>& lambda);
arma::vec get_params(std::vector<WeibNode*>& forest);
void UpdateHypers(WeibParams& weib_params,
                  std::vector<WeibNode*>& trees,
                  WeibData& data);

void UpdateScaleLambda(WeibParams& weib_params, const arma::vec& lambda);

#endif
