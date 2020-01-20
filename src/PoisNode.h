#ifndef POIS_NODE_H
#define POIS_NODE_H


#include <RcppArmadillo.h>
#include "Node.h"
#include "PoisParams.h"
#include "PoisData.h"
#include "PoisSS.h"

struct PoisNode : public Node<PoisNode> {

  double lambda;
  const PoisParams* pois_params;
  PoisSuffStats ss;

 PoisNode(TreeHypers* tree_hypers_, PoisParams* pois_params_) :
  Node<PoisNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 PoisNode(PoisNode* parent) : Node<PoisNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const PoisData& data, int i);
  void UpdateSuffStat(const PoisData& data);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(PoisNode* n, const arma::rowvec& x);
arma::vec PredictPois(PoisNode* tree, const arma::mat& X);

void BackFit(PoisData& data, PoisNode* tree);
void Refit(PoisData& data, PoisNode* tree);
double LogLT(PoisNode* root, const PoisData& data);
void UpdateParams(PoisNode* root, const PoisData& data);

#endif
