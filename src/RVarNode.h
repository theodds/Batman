#ifndef RVAR_NODE_H
#define RVAR_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "RVarParams.h"
#include "RVarData.h"
#include "RVarSS.h"

struct RVarNode : public Node<RVarNode> {
  
  double tau;
  const RVarParams* var_params;
  RVarSuffStats ss;

 RVarNode(TreeHypers* tree_hypers_, RVarParams* var_params_) :
  Node<RVarNode>(tree_hypers_), ss() {
    tau = 1.0;
    var_params = var_params_;
  }

 RVarNode(RVarNode* parent) : Node<RVarNode>(parent), ss() {
    tau = 1.0;
    var_params = parent->var_params;
  }

  void AddSuffStat(const RVarData& data, int i);
  void UpdateSuffStat(const RVarData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictVar(RVarNode* tree, const arma::rowvec& x);
arma::vec PredictVar(RVarNode* tree, const arma::mat& X);
void BackFit(RVarData& data, RVarNode* tree);
void Refit(RVarData& data, RVarNode* tree);
double LogLT(RVarNode* root, const RVarData& data);
void UpdateParams(RVarNode* root, const RVarData& data);

#endif
