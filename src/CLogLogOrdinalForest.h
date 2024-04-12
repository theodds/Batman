#ifndef CLOGLOGORDINAL_FOREST_H
#define CLOGLOGORDINAL_FOREST_H

#include <RcppArmadillo.h>
#include "CLogLogOrdinalNode.h"
#include "CLogLogOrdinalData.h"
#include "mcmc.h"


struct CLogLogOrdinalForest {
  std::vector<CLogLogOrdinalNode*> trees;

  CLogLogOrdinalForest(int num_trees,
                       TreeHypers* tree_hypers,
                       CLogLogOrdinalParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      CLogLogOrdinalNode* n = new CLogLogOrdinalNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~CLogLogOrdinalForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<CLogLogOrdinalNode*>& forest,
                      const arma::mat& X);

void UpdateHypers(CLogLogOrdinalParams& hypers,
                  std::vector<CLogLogOrdinalNode*>& trees,
                  const CLogLogOrdinalData& data);

#endif
