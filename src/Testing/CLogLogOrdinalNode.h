#ifndef CLOGLOGORDINAL_NODE_H
#define CLOGLOGORDINAL_NODE_H

// #include <RcppArmadillo.h>
// #include "Node.h"
// #include "GammaRegParams.h"
// #include "GammaRegData.h"
// #include "GammaRegSS.h"

// struct GammaNode : public Node<GammaNode> {

//   double lambda;
//   const GammaParams* pois_params;
//   GammaSuffStats ss;

//  GammaNode(TreeHypers* tree_hypers_, GammaParams* pois_params_) :
//   Node<GammaNode>(tree_hypers_), ss() {
//     lambda = 0.0;
//     pois_params = pois_params_;
//   }

//  GammaNode(GammaNode* parent) : Node<GammaNode>(parent), ss() {
//     lambda = 0.0;
//     pois_params = parent->pois_params;
//   }

//   void AddSuffStat(const GammaData& data, int i, double phi);
//   void UpdateSuffStat(const GammaData& data, double phi);

//   void ResetSuffStat() {
//     ss.Reset();
//     if(!is_leaf) {
//       left->ResetSuffStat();
//       right->ResetSuffStat();
//     }
//   }
// };

// double PredictPois(GammaNode* n, const arma::rowvec& x);
// arma::vec PredictPois(GammaNode* tree, const arma::mat& X);

// void BackFit(GammaData& data, GammaNode* tree);
// void Refit(GammaData& data, GammaNode* tree);
// double LogLT(GammaNode* root, const GammaData& data);
// void UpdateParams(GammaNode* root, const GammaData& data);

#endif
