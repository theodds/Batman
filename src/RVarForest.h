#ifndef VAR_FOREST_H
#define VAR_FOREST_H

#include <RcppArmadillo.h>
#include "VarNode.h"
#include "VarParams.h"
#include "mcmc.h"

struct VarForest {
  std::vector<VarNode*> trees;

  VarForest(int num_trees, TreeHypers* tree_hypers_, VarParams* var_params_)
  {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      VarNode* n = new VarNode(tree_hypers_, var_params_);
      trees.push_back(n);
    }
  }

  ~VarForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::mat PredictVar(std::vector<VarNode*>& forest, const arma::mat& X);
void UpdateHypers(VarParams& var_params,
                  std::vector<VarNode*>& trees,
                  const VarData& data);
void get_params(VarNode* n, std::vector<double>& mu, std::vector<double>& tau);
void UpdateTau0(VarParams& var_params, VarData& data);
void UpdateKappa(VarParams& var_params,
            std::vector<double>& mu,
            std::vector<double>& tau);
void UpdateScaleLogTau(VarParams& var_params, std::vector<double>& tau);
// void UpdateScaleLambda(MLogitParams& mlogit_params,
//                        std::vector<MLogitNode*>& trees);

#endif
