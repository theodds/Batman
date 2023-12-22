#ifndef QMNOM_FOREST_H
#define QMNOM_FOREST_H

#include <RcppArmadillo.h>
#include "QMultinomNode.h"
#include "QMultinomData.h"
#include "mcmc.h"


struct QMultinomForest {
  std::vector<QMultinomNode*> trees;

  QMultinomForest(int num_trees,
                  TreeHypers* tree_hypers,
                  QMultinomParams* mnom_params,
                  int K) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QMultinomNode* n = new QMultinomNode(tree_hypers, mnom_params, K);
      trees.push_back(n);
    }
  }

  ~QMultinomForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }
};

arma::mat Predict(std::vector<QMultinomNode*>& forest, const arma::mat& X);

void UpdateHypers(QMultinomParams& hypers, std::vector<QMultinomNode*>& trees,
                  QMultinomData& data);

#endif
