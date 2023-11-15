#ifndef RVAR_FOREST_H
#define RVAR_FOREST_H

#include <RcppArmadillo.h>
#include "RVarNode.h"
#include "RVarParams.h"
#include "mcmc.h"

struct RVarForest {
  std::vector<RVarNode*> trees;

  RVarForest(int num_trees, TreeHypers* tree_hypers_, RVarParams* var_params_)
  {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      VarNode* n = new RVarNode(tree_hypers_, var_params_);
      trees.push_back(n);
    }
  }

  ~RVarForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::vec PredictVar(std::vector<RVarNode*>& forest, const arma::mat& X);
void UpdateHypers(RVarParams& var_params,
                  std::vector<RVarNode*>& trees,
                  const RVarData& data);
void get_params(RVarNode* n, std::vector<double>& tau);
void UpdateTau0(RVarParams& var_params, RVarData& data);
void UpdateScaleLogTau(RVarParams& var_params, std::vector<double>& tau);

#endif
