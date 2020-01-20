#ifndef POIS_FOREST_H
#define POIS_FOREST_H

#include <RcppArmadillo.h>
#include "PoisNode.h"
#include "PoisData.h"
#include "mcmc.h"


struct PoisForest {
  std::vector<PoisNode*> trees;

  PoisForest(int num_trees, TreeHypers* tree_hypers, PoisParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      PoisNode* n = new PoisNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~PoisForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<PoisNode*>& forest, const arma::mat& X);

void UpdateHypers(PoisParams& hypers, std::vector<PoisNode*>& trees,
                  const PoisData& data);

#endif
