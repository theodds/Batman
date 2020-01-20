#ifndef WEIB_NODE_H
#define WEIB_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "WeibParams.h"
#include "WeibData.h"
#include "WeibSS.h"

struct WeibNode : public Node<WeibNode> {

  double lambda;
  const WeibParams* weib_params;
  WeibSuffStats ss;

  WeibNode(TreeHypers* tree_hypers_,
           WeibParams* weib_params_) :
  Node<WeibNode>(tree_hypers_), ss() {
    lambda = 0.0;
    weib_params = weib_params_;
  }

 WeibNode(WeibNode* parent) : Node<WeibNode>(parent), ss() {
    lambda = 0.0;
    weib_params = parent->weib_params;
  }

  void AddSuffStat(const arma::rowvec& x,
                   double y_elam,
                   double num_w,
                   double sum_log_w,
                   double lam_num_w);
  void UpdateSuffStat(const WeibData& data, double weibull_power);

  void ResetSuffStat() {
    ss.Reset();
    if(!is_leaf) {
      left->ResetSuffStat();
      right->ResetSuffStat();
    }
  }
};

double PredictWeib(WeibNode* tree, const arma::rowvec& x);
arma::vec PredictWeib(WeibNode* tree, const arma::mat& X);
void BackFit(WeibData& data, WeibNode* tree);
void Refit(WeibData& data, WeibNode* tree);
double LogLT(WeibNode* root, const WeibData& data);
void UpdateParams(WeibNode* root, const WeibData& data);


#endif
