#ifndef M_LOGIT_FOREST_H
#define M_LOGIT_FOREST_H

#include <RcppArmadillo.h>
#include "MLogitNode.h"
#include "MLogitParams.h"
#include "mcmc.h"

struct MLogitForest {
  std::vector<MLogitNode*> trees;

  MLogitForest(int num_trees,
               TreeHypers* tree_hypers_,
               MLogitParams* mlogit_hypers_)
  {
    trees.resize(0);
    for(int t = 0; t < num_trees; t++) {
      MLogitNode* n = new MLogitNode(tree_hypers_, 
                                     mlogit_hypers_, 
                                     mlogit_hypers_->lambda_0.size());
      trees.push_back(n);
    }
  }

  ~MLogitForest() {
    for(int t = 0; t < trees.size(); t++)
      delete trees[t];
  }
};

arma::mat PredictMLogit(std::vector<MLogitNode*>& forest, const arma::mat& X); 
void UpdateHypers(MLogitParams& mlogit_params,
                  std::vector<MLogitNode*>& trees,
                  const MLogitData& data);
arma::vec get_params(std::vector<MLogitNode*>& forest);
void get_params(MLogitNode* n, std::vector<double>& mu);
void UpdateScaleLambda(MLogitParams& mlogit_params,
                       std::vector<MLogitNode*>& trees);

void UpdateLambda0(MLogitParams& mlogit_params, MLogitData& data);
void UpdateScaleLambda(MLogitParams& mlogit_params,
                       const arma::vec& lambda);
arma::mat LambdaToPi(const arma::mat& lambda);

#endif
