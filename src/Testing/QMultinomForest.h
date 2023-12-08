#ifndef QBINOM_FOREST_H
#define QBINOM_FOREST_H

#include <RcppArmadillo.h>
#include "QBinomNode.h"
#include "QBinomData.h"
#include "mcmc.h"


struct QBinomForest {
  std::vector<QBinomNode*> trees;

  QBinomForest(int num_trees, TreeHypers* tree_hypers, QBinomParams* pois_params) {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      QBinomNode* n = new QBinomNode(tree_hypers, pois_params);
      trees.push_back(n);
    }
  }

  ~QBinomForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }

};

arma::vec PredictPois(std::vector<QBinomNode*>& forest, const arma::mat& X);

void UpdateHypers(QBinomParams& hypers, std::vector<QBinomNode*>& trees,
                  QBinomData& data);

#endif
