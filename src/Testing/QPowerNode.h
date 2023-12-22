#ifndef QPOWER_NODE_H
#define QPOWER_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QPowerParams.h"
#include "QPowerData.h"
#include "QPowerSS.h"

struct QPowerNode : public Node<QPowerNode> {

  double lambda;
  const QPowerParams* pois_params;
  QPowerSuffStats ss;

 QPowerNode(TreeHypers* tree_hypers_, QPowerParams* pois_params_) :
  Node<QPowerNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 QPowerNode(QPowerNode* parent) : Node<QPowerNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const QPowerData& data, int i, double phi, double p);
  void UpdateSuffStat(const QPowerData& data, double phi, double p);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(QPowerNode* n, const arma::rowvec& x);
arma::vec PredictPois(QPowerNode* tree, const arma::mat& X);

void BackFit(QPowerData& data, QPowerNode* tree);
void Refit(QPowerData& data, QPowerNode* tree);
double LogLT(QPowerNode* root, const QPowerData& data);
void UpdateParams(QPowerNode* root, const QPowerData& data);

#endif
