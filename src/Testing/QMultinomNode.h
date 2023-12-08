#ifndef QMNOM_NODE_H
#define QMNOM_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "QMultinomParams.h"
#include "QMultinomData.h"
#include "QMultinomSS.h"

struct QMultinomNode : public Node<QMultinomNode> {

  arma::vec lambda;
  const QMultinomParams* mnom_params;
  QMultinomSuffStats ss;

  QMultinomNode(TreeHypers* tree_hypers_, QMultinomParams* mnom_params_, int K) :
  Node<QMultinomNode>(tree_hypers_), ss() {
    lambda                         = arma::zeros<arma::vec>(K);
    ss.sum_Y_by_phi                = arma::zeros<arma::vec>(K);
    ss.sum_exp_lambda_minus_by_phi = arma::zeros<arma::vec>(K);
    mnom_params                    = mnom_params_;
  }

  QMultinomNode(QMultinomNode* parent) : Node<QMultinomNode>(parent), ss() {
    int K = parent->lambda.n_elem;
    lambda = arma::zeros<arma::vec>(K);
    ss.sum_Y_by_phi                = arma::zeros<arma::vec>(K);
    ss.sum_exp_lambda_minus_by_phi = arma::zeros<arma::vec>(K);
    pois_params = parent->pois_params;
 }

  void AddSuffStat(const QMultinomData& data, int i, double phi);
  void UpdateSuffStat(const QMultinomData& data, double phi);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

arma::vec Predict(QMultinomNode* n, const arma::rowvec& x);
arma::mat Predict(QMultinomNode* tree, const arma::mat& X);

void BackFit(QMultinomData& data, QMultinomNode* tree);
void Refit(QMultinomData& data, QMultinomNode* tree);
double LogLT(QMultinomNode* root, const QMultinomData& data);
void UpdateParams(QMultinomNode* root, const QMultinomData& data);

#endif
