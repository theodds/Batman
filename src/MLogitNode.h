#ifndef M_LOGIT_NODE_H
#define M_LOGIT_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "MLogitParams.h"
#include "MLogitData.h"
#include "MLogitSS.h"

struct MLogitNode : public Node<MLogitNode> {

  arma::vec lambda;
  const MLogitParams* mlogit_params;
  MLogitSuffStats ss;

 MLogitNode(TreeHypers* tree_hypers_,
            MLogitParams* mlogit_params_,
            int num_cat) :
  Node<MLogitNode>(tree_hypers_), ss(num_cat) {
    lambda = arma::zeros<arma::vec>(num_cat);
    mlogit_params = mlogit_params_;
  }

 MLogitNode(MLogitNode* parent) :
  Node<MLogitNode>(parent),
    ss(parent->lambda.size()) {
    lambda        = arma::zeros<arma::vec>(parent->lambda.size());
    mlogit_params = parent->mlogit_params;
  }

  void AddSuffStat(const MLogitData& data,int i,const arma::vec& exp_lambda_hat);
  void UpdateSuffStat(const MLogitData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

arma::vec PredictMLogit(MLogitNode* tree, const arma::rowvec& x);
arma::mat PredictMLogit(MLogitNode* tree, const arma::mat& X);
void BackFit(MLogitData& data, MLogitNode* tree);
void Refit(MLogitData& data, MLogitNode* tree);
double LogLT(MLogitNode* root, const MLogitData& data);
void UpdateParams(MLogitNode* root, const MLogitData& data);

#endif

