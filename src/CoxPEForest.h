#ifndef COX_PE_FOREST_H
#define COX_PE_FOREST_H

#include <RcppArmadillo.h>
#include "CoxPENode.h"
#include "CoxPEParams.h"
#include "mcmc.h"

struct CoxPEForest {
  std::vector<CoxPENode*> trees;

  CoxPEForest(int num_trees, TreeHypers* tree_hypers, CoxPEParams* cox_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      CoxPENode* n = new CoxPENode(tree_hypers, cox_params);
      trees.push_back(n);
    }
  }

  ~CoxPEForest() {
    for(int t = 0; t < trees.size(); t++) {
      delete trees[t];
    }
  }
};

arma::vec PredictCox(std::vector<CoxPENode*>& forest, const arma::mat& X);
void UpdateHypers(CoxPEParams& cox_params,
                  std::vector<CoxPENode*>& trees,
                  CoxPEData& data);
void get_params(CoxPENode* n, std::vector<double>& lambda);
void UpdateScaleLambda(CoxPEParams& cox_params, std::vector<double>& lambda);
/* void UpdatePEBase(CoxData& data); */

#endif
