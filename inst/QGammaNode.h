#ifndef QGamma_NODE_H
#define QGamma_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QGammaParams.h"
#include "QGammaData.h"
#include "QGammaSS.h"

struct QGammaNode : public Node<QGammaNode> {

  double lambda;
  const QGammaParams* pois_params;
  QGammaSuffStats ss;

 QGammaNode(TreeHypers* tree_hypers_, QGammaParams* pois_params_) :
  Node<QGammaNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 QGammaNode(QGammaNode* parent) : Node<QGammaNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const QGammaData& data, int i, double phi);
  void UpdateSuffStat(const QGammaData& data, double phi);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(QGammaNode* n, const arma::rowvec& x);
arma::vec PredictPois(QGammaNode* tree, const arma::mat& X);

void BackFit(QGammaData& data, QGammaNode* tree);
void Refit(QGammaData& data, QGammaNode* tree);
double LogLT(QGammaNode* root, const QGammaData& data);
void UpdateParams(QGammaNode* root, const QGammaData& data);

#endif
