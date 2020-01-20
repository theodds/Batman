#ifndef VAR_LOGIT_NODE_H
#define VAR_LOGIT_NODE_H

#include "VarNode.h"
#include "MLogitNode.h"
#include "VarLogitParams.h"
#include "VarLogitData.h"
#include "VarLogitSS.h"

struct VarLogitNode : public Node<VarLogitNode> {
  
  double tau;
  double mu;
  arma::vec lambda;
  VarLogitParams* params;
  VarLogitSuffStats ss;

 VarLogitNode(TreeHypers* tree_hypers_, VarLogitParams* params_, int num_cat) :
  Node<VarLogitNode>(tree_hypers_), ss(num_cat) {
    lambda = arma::zeros<arma::vec>(num_cat);
    tau = 1.;
    mu = 0.;
    params = params_;
  }

 VarLogitNode(VarLogitNode* parent) : Node<VarLogitNode>(parent),
    ss(parent->lambda.size()) {
    lambda = arma::zeros<arma::vec>(parent->lambda.size());
    params = parent->params;
    tau = 1.;
    mu = 0.;
  }

  void AddSuffStatLogit(const MLogitData& data, 
                        int i,
                        const arma::vec& exp_lambda_hat);
  void AddSuffStatVar(const VarData& data, int i);
  void UpdateSuffStat(const VarData& data_var, const MLogitData& data_logit);
  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }

};

arma::vec PredictMLogit(VarLogitNode* tree, const arma::rowvec& x); 
arma::mat PredictMLogit(VarLogitNode* tree, const arma::mat& X);
arma::vec PredictVar(VarLogitNode* tree, const arma::rowvec& x);
arma::mat PredictVar(VarLogitNode* tree, const arma::mat& X);
void BackFit(VarLogitData& data, VarLogitNode* tree);
void Refit(VarLogitData& data, VarLogitNode* tree);
double LogLT(VarLogitNode* root, const VarLogitData& data);
void UpdateParams(VarLogitNode* root, const VarLogitData& data);

#endif
