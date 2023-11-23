#ifndef QGAMMA_FOREST_H
#define QGAMMA_FOREST_H

#include <RcppArmadillo.h>
#include "QGammaNode.h"
#include "QGammaData.h"
#include "mcmc.h"


struct QGammaForest {
  std::vector<QGammaNode*> trees;

  QGammaForest(int num_trees, TreeHypers* tree_hypers, QGammaParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QGammaNode* n = new QGammaNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QGammaForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<QGammaNode*>& forest, const arma::mat& X);

void UpdateHypers(QGammaParams& hypers, std::vector<QGammaNode*>& trees,
                  const QGammaData& data);

#endif
