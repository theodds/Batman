#ifndef REG_NODE_H
#define REG_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "RegParams.h"
#include "RegData.h"
#include "RegSS.h"

struct RegNode : public Node<RegNode> {

  double mu;
  const RegParams* reg_params;
  RegSuffStats ss;


  RegNode(TreeHypers* tree_hypers_,
          RegParams* reg_params_) : Node<RegNode>(tree_hypers_), ss() {
    mu = 0.0;
    reg_params = reg_params_;
  }

  RegNode(RegNode* parent) : Node<RegNode>(parent), ss() {
    mu = 0.0;
    reg_params = parent->reg_params;
  }

  void AddSuffStat(const RegData& data, int i);
  void UpdateSuffStat(const RegData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }

};

double PredictReg(RegNode* n, const arma::rowvec& x);
arma::vec PredictReg(RegNode* tree, const arma::mat& X);

void BackFit(RegData& data, RegNode* tree);
void Refit(RegData& data, RegNode* tree);
double LogLT(RegNode* root, const RegData& data);
void UpdateParams(RegNode* root, const RegData& data);


#endif
