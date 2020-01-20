#ifndef VAR_LOGIT_FOREST_H
#define VAR_LOGIT_FOREST_H

#include <RcppArmadillo.h>
#include "VarLogitNode.h"
#include "VarLogitParams.h"
#include "mcmc.h"
#include "MLogitForest.h"
#include "VarForest.h"

struct VarLogitForest {
  std::vector<VarLogitNode*> trees;

  VarLogitForest(int num_trees, TreeHypers* tree_hypers, VarLogitParams* params, int num_cat)
  {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      VarLogitNode* n = new VarLogitNode(tree_hypers, params, num_cat);
      trees.push_back(n);
    }
  }

  ~VarLogitForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};


arma::mat PredictMLogit(std::vector<VarLogitNode*>& forest, const arma::mat& X); 
arma::mat PredictVar(std::vector<VarLogitNode*>& forest, const arma::mat& X); 
void UpdateHypers(VarLogitParams& params,
                  std::vector<VarLogitNode*>& trees,
                  VarLogitData& data);
void UpdatePhi(VarLogitData& data, VarLogitParams& mlogit_params);
void get_mu_tau(std::vector<VarLogitNode*> trees, std::vector<double>& mu, std::vector<double>& tau);
void get_lambda(std::vector<VarLogitNode*> trees, std::vector<double>& lambda);



#endif
