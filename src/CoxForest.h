#ifndef COX_FOREST_H
#define COX_FOREST_H

#include <RcppArmadillo.h>
#include "CoxNode.h"
#include "CoxParams.h"
#include "mcmc.h"

struct CoxForest {
  std::vector<CoxNode*> trees;

  CoxForest(int num_trees, TreeHypers* tree_hypers, CoxParams* cox_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      CoxNode* n = new CoxNode(tree_hypers, cox_params);
      trees.push_back(n);
    }
  }

  ~CoxForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::vec PredictCox(std::vector<CoxNode*>& forest, const arma::mat& X);
void UpdateHypers(CoxParams& cox_params,
                  std::vector<CoxNode*>& trees,
                  CoxData& data);
void get_params(CoxNode* n, std::vector<double>& lambda);
void UpdateScaleLambda(CoxParams& cox_params, std::vector<double>& lambda);
void UpdatePhi(CoxData& data);

#endif
