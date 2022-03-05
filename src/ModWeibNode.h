#ifndef MOD_WEIB_NODE_H
#define MOD_WEIB_NODE_H

#include <RcppArmadillo.h>
#include "Node.h"
#include "ModWeibParams.h"
#include "ModWeibData.h"
#include "ModWeibSS.h"

struct ModWeibNode : public Node<ModWeibNode> {

    // Node Params
    double mu;
    double frequency;
    double offset;

    // Pointer to params
    const ModWeibParams* mod_weib_params;
    ModWeibSuffStats ss;

    ModWeibNode(TreeHypers* tree_hypers_,
                ModWeibParams* mod_weib_params_) 
                : Node<ModWeibNode>(tree_hypers_), ss() {
      mu = 0.0
      mod_weib_params = mod_weib_params_;
      frequency = norm_rand() / hypers.length_scale;
      offset = 2. * M_PI * unif_rand();
  }

  ModWeibNode(ModWeibNode* parent) : Node<ModWeibNode>(parent), ss() {
      mu = 0.0;
      mod_weib_params = parent->mod_weib_params;
      frequency = parent->frequency;
      offset = parent->offset;
  }

};



//   void AddSuffStat(const arma::rowvec& x,
//                    double y_elam,
//                    double num_w,
//                    double sum_log_w,
//                    double lam_num_w);
//   void UpdateSuffStat(const WeibData& data, double weibull_power);

//   void ResetSuffStat() {
//     ss.Reset();
//     if(!is_leaf) {
//       left->ResetSuffStat();
//       right->ResetSuffStat();
//     }
//   }
// };

// double PredictWeib(WeibNode* tree, const arma::rowvec& x);
// arma::vec PredictWeib(WeibNode* tree, const arma::mat& X);
// void BackFit(WeibData& data, WeibNode* tree);
// void Refit(WeibData& data, WeibNode* tree);
// double LogLT(WeibNode* root, const WeibData& data);
// void UpdateParams(WeibNode* root, const WeibData& data);

#endif