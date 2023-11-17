#ifndef QPOIS_NODE_H
#define QPOIS_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QPoisParams.h"
#include "QPoisData.h"
#include "QPoisSS.h"

struct QPoisNode : public Node<QPoisNode> {

  double lambda;
  const QPoisParams* pois_params;
  QPoisSuffStats ss;

 QPoisNode(TreeHypers* tree_hypers_, QPoisParams* pois_params_) :
  Node<QPoisNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 QPoisNode(QPoisNode* parent) : Node<QPoisNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const QPoisData& data, int i, double phi);
  void UpdateSuffStat(const QPoisData& data, double phi);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(QPoisNode* n, const arma::rowvec& x);
arma::vec PredictPois(QPoisNode* tree, const arma::mat& X);

void BackFit(QPoisData& data, QPoisNode* tree);
void Refit(QPoisData& data, QPoisNode* tree);
double LogLT(QPoisNode* root, const QPoisData& data);
void UpdateParams(QPoisNode* root, const QPoisData& data);

#endif
