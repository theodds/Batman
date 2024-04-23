#ifndef COX_NPHO_NODE_H
#define COX_NPHO_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "CoxNPHOParams.h"
#include "CoxNPHOData.h"
#include "CoxNPHOSS.h"

struct CoxNPHONode : public Node<CoxNPHONode> {

  double lambda;
  const CoxNPHOParams* cox_params;
  CoxNPHOSuffStats ss;

  CoxNPHONode(TreeHypers* tree_hypers_, CoxNPHOParams* cox_params_)
    : Node<CoxNPHONode>(tree_hypers_), ss()
    {
      lambda = 0.;
      cox_params = cox_params_;
    }

 CoxNPHONode(CoxNPHONode* parent) : Node<CoxNPHONode>(parent), ss()
    {
      lambda = 0.;
      cox_params = parent->cox_params;
    }

  void AddSuffStat(double delta_b,
                   double Z,
                   double lambda_minus,
                   double gamma,
                   double b_float,
                   const arma::rowvec& x);
  void UpdateSuffStat(const CoxNPHOData& data, const arma::vec& gamma);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

arma::rowvec PredictCox(CoxNPHONode* n, const arma::rowvec& x, int K);
arma::mat PredictCox(CoxNPHONode* tree, const arma::mat& X, int K);
double PredictCox(CoxNPHONode* n, const arma::rowvec& x, int K, 
                  double b_float);

void BackFit(CoxNPHOData& data, CoxNPHONode* tree);
void Refit(CoxNPHOData& data, CoxNPHONode* tree);
double LogLT(CoxNPHONode* root, const CoxNPHOData& data);
void UpdateParams(CoxNPHONode* root, const CoxNPHOData& data);

#endif
