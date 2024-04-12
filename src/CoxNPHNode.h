#ifndef COX_NPH_NODE_H
#define COX_NPH_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "CoxNPHParams.h"
#include "CoxNPHData.h"
#include "CoxNPHSS.h"

struct CoxNPHNode : public Node<CoxNPHNode> {

  double lambda;
  const CoxNPHParams* cox_params;
  CoxNPHSuffStats ss;

  CoxNPHNode(TreeHypers* tree_hypers_, CoxNPHParams* cox_params_)
    : Node<CoxNPHNode>(tree_hypers_), ss()
    {
      lambda = 0.;
      cox_params = cox_params_;
    }

 CoxNPHNode(CoxNPHNode* parent) : Node<CoxNPHNode>(parent), ss()
    {
      lambda = 0.;
      cox_params = parent->cox_params;
    }

  void AddSuffStat(double delta_b,
                   double lambda_minus,
                   double Z, 
                   double base_haz,
                   double b_float,
                   const arma::rowvec& x);
  void UpdateSuffStat(const CoxNPHData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

arma::rowvec PredictCox(CoxNPHNode* n, const arma::rowvec& x, int num_bin);
arma::mat PredictCox(CoxNPHNode* tree, const arma::mat& X, int num_bin);
double PredictCox(CoxNPHNode* n, const arma::rowvec& x, int num_bin, 
                  double b_float);

void BackFit(CoxNPHData& data, CoxNPHNode* tree);
void Refit(CoxNPHData& data, CoxNPHNode* tree);
double LogLT(CoxNPHNode* root, const CoxNPHData& data);
void UpdateParams(CoxNPHNode* root, const CoxNPHData& data);

#endif
