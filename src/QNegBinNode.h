#ifndef QNB_NODE_H
#define QNB_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QNegBinParams.h"
#include "QNegBinData.h"
#include "QNegBinSS.h"

struct QNBNode : public Node<QNBNode> {

  double lambda;
  const QNBParams* pois_params;
  QNBSuffStats ss;

 QNBNode(TreeHypers* tree_hypers_, QNBParams* pois_params_) :
  Node<QNBNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 QNBNode(QNBNode* parent) : Node<QNBNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const QNBData& data, int i, double phi);
  void UpdateSuffStat(const QNBData& data, double phi);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(QNBNode* n, const arma::rowvec& x);
arma::vec PredictPois(QNBNode* tree, const arma::mat& X);

void BackFit(QNBData& data, QNBNode* tree);
void Refit(QNBData& data, QNBNode* tree);
double LogLT(QNBNode* root, const QNBData& data);
void UpdateParams(QNBNode* root, const QNBData& data);

#endif
