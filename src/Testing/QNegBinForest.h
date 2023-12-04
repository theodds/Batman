#ifndef QPOIS_FOREST_H
#define QPOIS_FOREST_H

#include <RcppArmadillo.h>
#include "QPoisNode.h"
#include "QPoisData.h"
#include "mcmc.h"


struct QPoisForest {
  std::vector<QPoisNode*> trees;

  QPoisForest(int num_trees, TreeHypers* tree_hypers, QPoisParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QPoisNode* n = new QPoisNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QPoisForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<QPoisNode*>& forest, const arma::mat& X);

void UpdateHypers(QPoisParams& hypers, std::vector<QPoisNode*>& trees,
                  const QPoisData& data);

#endif
