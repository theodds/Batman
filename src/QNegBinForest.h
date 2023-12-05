#ifndef QNB_FOREST_H
#define QNB_FOREST_H

#include <RcppArmadillo.h>
#include "QNegBinNode.h"
#include "QNegBinData.h"
#include "mcmc.h"


struct QNBForest {
  std::vector<QNBNode*> trees;

  QNBForest(int num_trees, TreeHypers* tree_hypers, QNBParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QNBNode* n = new QNBNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QNBForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<QNBNode*>& forest, const arma::mat& X);

void UpdateHypers(QNBParams& hypers, std::vector<QNBNode*>& trees,
                  const QNBData& data);

#endif
