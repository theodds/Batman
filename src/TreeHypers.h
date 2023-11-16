#ifndef TREE_HYPERS_H
#define TREE_HYPERS_H

#include <RcppArmadillo.h>
#include "functions.h"
#include "slice.h"

struct TreeHypers {

  double alpha;
  double beta;
  double gamma;

  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  double get_s(int j){return s(j);}
  arma::vec get_s(){return s;}
  double get_log_s(int j){return logs(j);}
  int get_num_groups() const {return num_groups;}
  void set_s(arma::vec x) {s = x; logs = log(x);}
  void set_log_s(arma::vec x) {s = exp(x); logs = x;}

  TreeHypers(const arma::sp_mat& probs_);
  arma::uvec SampleVar() const;
  void UpdateAlpha();

  bool update_alpha;
  bool update_s;

private:

  arma::vec s; // logs = log(s)
  arma::vec logs;
  int num_groups;
  arma::sp_mat probs; // Columns sum to 1

};


double GrowProb(const TreeHypers* tree_hypers, int depth);

#endif
