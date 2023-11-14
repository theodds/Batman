#ifndef VAR_NODE_H
#define VAR_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "VarParams.h"
#include "VarData.h"
#include "VarSS.h"

struct VarNode : public Node<VarNode> {
  
  double tau;
  double mu;
  const VarParams* var_params;
  VarSuffStats ss;

 VarNode(TreeHypers* tree_hypers_, VarParams* var_params_) :
  Node<VarNode>(tree_hypers_), ss() {
    tau = 1.0;
    mu = 0.0;
    var_params = var_params_;
  }

 VarNode(VarNode* parent) : Node<VarNode>(parent), ss() {
    tau = 1.0;
    mu = 0.0;
    var_params = parent->var_params;
  }

  void AddSuffStat(const VarData& data, int i);
  void UpdateSuffStat(const VarData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

arma::vec PredictVar(VarNode* tree, const arma::rowvec& x);
arma::mat PredictVar(VarNode* tree, const arma::mat& X);
void BackFit(VarData& data, VarNode* tree);
void Refit(VarData& data, VarNode* tree);
double LogLT(VarNode* root, const VarData& data);
void UpdateParams(VarNode* root, const VarData& data);

#endif
