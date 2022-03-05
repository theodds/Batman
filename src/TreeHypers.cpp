#include "TreeHypers.h"

using namespace Rcpp;
using namespace arma;

TreeHypers::TreeHypers(const arma::sp_mat& probs_) {

  // Init s
  probs      = probs_;
  num_groups = probs.n_cols;
  s          = ones<vec>(num_groups) / ((double)(num_groups));
  logs       = log(s);

  // Init non-s members
  alpha         = 1.0;
  beta          = 2.0;
  gamma         = 0.95;
  alpha_scale   = ((double)(num_groups));
  alpha_shape_1 = 0.5;
  alpha_shape_2 = 1.0;

  update_s = true;
  update_alpha = true;

}

arma::uvec TreeHypers::SampleVar() const {
  uvec group_var = zeros<uvec>(2);
  group_var(0) = sample_class(s);
  group_var(1) = sample_class_col(probs, group_var(0));
  return group_var;
}

double GrowProb(const TreeHypers* tree_hypers, int depth) {
  double d = (double)(depth);
  return tree_hypers->gamma * pow(1.0 + d, -tree_hypers->beta);
}

void TreeHypers::UpdateAlpha() {
  RhoLoglik* loglik = new RhoLoglik(mean(logs), (double)s.size(), alpha_scale,
                                    alpha_shape_1, alpha_shape_2);

  double rho_current = alpha / (alpha + alpha_scale);
  double rho_up = slice_sampler(rho_current, loglik, 0.1, 0.0, 1.0);

  alpha = alpha_scale * rho_up / (1.0 - rho_up);

  delete loglik;
}
