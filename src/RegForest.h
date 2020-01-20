#ifndef REG_FOREST_H
#define REG_FOREST_H

#include <RcppArmadillo.h>
#include "RegNode.h"
#include "RegData.h"
#include "mcmc.h"

struct RegForest {
  std::vector<RegNode*> trees;

  RegForest(int num_trees, TreeHypers* tree_hypers, RegParams* reg_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      RegNode* n = new RegNode(tree_hypers, reg_params);
      trees.push_back(n);
    }
  }

  ~RegForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictReg(std::vector<RegNode*>& forest, const arma::mat& X);
void UpdateHypers(RegParams& hypers, std::vector<RegNode*>& trees, const RegData& data);
void UpdateSigmaY(RegParams& hypers, const RegData& data);
void UpdateSigmaMu(RegParams& hypers, std::vector<RegNode*>& forest);

#endif
