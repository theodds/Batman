#ifndef NPHSBP_FOREST_H
#define NPHSBP_FOREST_H

#include <RcppArmadillo.h>
#include "NPHSBPNode.h"
#include "NPHSBPParams.h"
#include "mcmc.h"

struct NPHSBPForest {
  std::vector<NPHSBPNode*> trees;

  NPHSBPForest(int num_trees, TreeHypers* tree_hypers, NPHSBPParams* cox_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      NPHSBPNode* n = new NPHSBPNode(tree_hypers, cox_params);
      trees.push_back(n);
    }
  }

  ~NPHSBPForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::mat PredictCox(std::vector<NPHSBPNode*>& forest, const arma::mat& X, int K);
void UpdateHypers(NPHSBPParams& cox_params,
                  std::vector<NPHSBPNode*>& trees,
                  NPHSBPData& data);
void get_params(NPHSBPNode* n, std::vector<double>& lambda);
void UpdateScaleLambda(NPHSBPParams& cox_params, std::vector<double>& lambda);

#endif
