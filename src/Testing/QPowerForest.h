#ifndef QPOWER_FOREST_H
#define QPOWER_FOREST_H

#include <RcppArmadillo.h>
#include "QPowerNode.h"
#include "QPowerData.h"
#include "mcmc.h"


struct QPowerForest {
  std::vector<QPowerNode*> trees;

  QPowerForest(int num_trees, TreeHypers* tree_hypers, QPowerParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QPowerNode* n = new QPowerNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QPowerForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<QPowerNode*>& forest, const arma::mat& X);

void UpdateHypers(QPowerParams& hypers, std::vector<QPowerNode*>& trees,
                  const QPowerData& data);

#endif
