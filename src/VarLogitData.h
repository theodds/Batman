#ifndef VAR_LOGIT_DATA
#define VAR_LOGIT_DATA

#include "VarData.h"
#include "MLogitData.h"

struct VarLogitData {
  VarData var_data;
  MLogitData mlogit_data;

  VarLogitData(const arma::mat& X_logit,
               const arma::uvec& Y_logit,
               int num_class,
               const arma::mat& X_var,
               const arma::mat& Y_var) :
  mlogit_data(X_logit, Y_logit, num_class),
    var_data(X_var, Y_var) {;}

};

#endif
