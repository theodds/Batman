#ifndef COX_NPH_FOREST_H
#define COX_NPH_FOREST_H

#include <RcppArmadillo.h>
#include "CoxNPHNode.h"
#include "CoxNPHParams.h"
#include "mcmc.h"

struct CoxNPHForest {
  std::vector<CoxNPHNode*> trees;

  CoxNPHForest(int num_trees, TreeHypers* tree_hypers, CoxNPHParams* cox_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      CoxNPHNode* n = new CoxNPHNode(tree_hypers, cox_params);
      trees.push_back(n);
    }
  }

  ~CoxNPHForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::mat PredictCox(std::vector<CoxNPHNode*>& forest, const arma::mat& X, int num_bin);
void UpdateHypers(CoxNPHParams& cox_params,
                  std::vector<CoxNPHNode*>& trees,
                  CoxNPHData& data);
void get_params(CoxNPHNode* n, std::vector<double>& lambda);
void UpdateScaleLambda(CoxNPHParams& cox_params, std::vector<double>& lambda);
/* void UpdatePEBase(CoxData& data); */

#endif
