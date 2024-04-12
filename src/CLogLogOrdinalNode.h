#ifndef CLOGLOGORDINAL_NODE_H
#define CLOGLOGORDINAL_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "CLogLogOrdinalParams.h"
#include "CLogLogOrdinalData.h"
#include "CLogLogOrdinalSS.h"

struct CLogLogOrdinalNode : public Node<CLogLogOrdinalNode> {

  double lambda;
  const CLogLogOrdinalParams* pois_params;
  CLogLogOrdinalSuffStats ss;

 CLogLogOrdinalNode(TreeHypers* tree_hypers_, CLogLogOrdinalParams* pois_params_) :
  Node<CLogLogOrdinalNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 CLogLogOrdinalNode(CLogLogOrdinalNode* parent) :
  Node<CLogLogOrdinalNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const CLogLogOrdinalData& data,
                   int i,
                   const arma::vec& gamma,
                   const arma::vec& seg);
  void UpdateSuffStat(const CLogLogOrdinalData& data,
                      const arma::vec& gamma,
                      const arma::vec& seg);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(CLogLogOrdinalNode* n, const arma::rowvec& x);
arma::vec PredictPois(CLogLogOrdinalNode* tree, const arma::mat& X);

void BackFit(CLogLogOrdinalData& data, CLogLogOrdinalNode* tree);
void Refit(CLogLogOrdinalData& data, CLogLogOrdinalNode* tree);
double LogLT(CLogLogOrdinalNode* root, const CLogLogOrdinalData& data);
void UpdateParams(CLogLogOrdinalNode* root, const CLogLogOrdinalData& data);

#endif
