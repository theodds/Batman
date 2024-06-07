#ifndef COX_NPHO_FOREST_H
#define COX_NPHO_FOREST_H

#include <RcppArmadillo.h>
#include "CoxNPHONode.h"
#include "CoxNPHOParams.h"
#include "mcmc.h"

struct CoxNPHOForest {
  std::vector<CoxNPHONode*> trees;

  CoxNPHOForest(int num_trees, TreeHypers* tree_hypers, CoxNPHOParams* cox_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      CoxNPHONode* n = new CoxNPHONode(tree_hypers, cox_params);
      trees.push_back(n);
    }
  }

  ~CoxNPHOForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::mat PredictCox(std::vector<CoxNPHONode*>& forest, const arma::mat& X, int K);
void UpdateHypers(CoxNPHOParams& cox_params,
                  std::vector<CoxNPHONode*>& trees,
                  CoxNPHOData& data);
void get_params(CoxNPHONode* n, std::vector<double>& lambda);
void UpdateScaleLambda(CoxNPHOParams& cox_params, std::vector<double>& lambda);

#endif
