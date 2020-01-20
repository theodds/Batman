#ifndef COX_NODE_H
#define COX_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "CoxParams.h"
#include "CoxData.h"
#include "CoxSS.h"

struct CoxNode : public Node<CoxNode> {

  double lambda;
  const CoxParams* cox_params;
  CoxSuffStats ss;

  CoxNode(TreeHypers* tree_hypers_, CoxParams* cox_params_)
    : Node<CoxNode>(tree_hypers_), ss()
    {
      lambda = 0.;
      cox_params = cox_params_;
    }

 CoxNode(CoxNode* parent) : Node<CoxNode>(parent), ss()
    {
      lambda = 0.;
      cox_params = parent->cox_params;
    }

  void AddSuffStat(const CoxData& data, int i);
  void UpdateSuffStat(const CoxData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictCox(CoxNode* n, const arma::rowvec& x);
arma::vec PredictCox(CoxNode* tree, const arma::mat& X);

void BackFit(CoxData& data, CoxNode* tree);
void Refit(CoxData& data, CoxNode* tree);
double LogLT(CoxNode* root, const CoxData& data);
void UpdateParams(CoxNode* root, const CoxData& data);

#endif
