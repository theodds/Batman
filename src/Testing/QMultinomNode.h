#ifndef QBinom_NODE_H
#define QBinom_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QBinomParams.h"
#include "QBinomData.h"
#include "QBinomSS.h"

struct QBinomNode : public Node<QBinomNode> {

  double lambda;
  const QBinomParams* pois_params;
  QBinomSuffStats ss;

 QBinomNode(TreeHypers* tree_hypers_, QBinomParams* pois_params_) :
  Node<QBinomNode>(tree_hypers_), ss() {
    lambda = 0.0;
    pois_params = pois_params_;
  }

 QBinomNode(QBinomNode* parent) : Node<QBinomNode>(parent), ss() {
    lambda = 0.0;
    pois_params = parent->pois_params;
  }

  void AddSuffStat(const QBinomData& data, int i, double phi);
  void UpdateSuffStat(const QBinomData& data, double phi);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictPois(QBinomNode* n, const arma::rowvec& x);
arma::vec PredictPois(QBinomNode* tree, const arma::mat& X);

void BackFit(QBinomData& data, QBinomNode* tree);
void Refit(QBinomData& data, QBinomNode* tree);
double LogLT(QBinomNode* root, const QBinomData& data);
void UpdateParams(QBinomNode* root, const QBinomData& data);

#endif
