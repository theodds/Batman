#ifndef COX_NODE_H
#define COX_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "CoxPEParams.h"
#include "CoxPEData.h"
#include "CoxPESS.h"

struct CoxPENode : public Node<CoxPENode> {

  double lambda;
  const CoxPEParams* cox_params;
  CoxPESuffStats ss;

  CoxPENode(TreeHypers* tree_hypers_, CoxPEParams* cox_params_)
    : Node<CoxPENode>(tree_hypers_), ss()
    {
      lambda = 0.;
      cox_params = cox_params_;
    }

 CoxPENode(CoxPENode* parent) : Node<CoxPENode>(parent), ss()
    {
      lambda = 0.;
      cox_params = parent->cox_params;
    }

  void AddSuffStat(const CoxPEData& data, int i);
  void UpdateSuffStat(const CoxPEData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictCox(CoxPENode* n, const arma::rowvec& x);
arma::vec PredictCox(CoxPENode* tree, const arma::mat& X);

void BackFit(CoxPEData& data, CoxPENode* tree);
void Refit(CoxPEData& data, CoxPENode* tree);
double LogLT(CoxPENode* root, const CoxPEData& data);
void UpdateParams(CoxPENode* root, const CoxPEData& data);

#endif
