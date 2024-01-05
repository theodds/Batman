#ifndef GAMMA_FOREST_H
#define GAMMA_FOREST_H

#include <RcppArmadillo.h>
#include "GammaRegNode.h"
#include "GammaRegData.h"
#include "mcmc.h"


struct GammaForest {
  std::vector<GammaNode*> trees;

  GammaForest(int num_trees, TreeHypers* tree_hypers, GammaParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      GammaNode* n = new GammaNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~GammaForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<GammaNode*>& forest, const arma::mat& X);

void UpdateHypers(GammaParams& hypers, std::vector<GammaNode*>& trees,
                  const GammaData& data);

#endif
