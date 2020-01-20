#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma; 

// [[Rcpp::depends(RcppArmadillo)]]

double WeibCumHazard(double t, double rate, double shape)
{
  return pow(t, shape) * rate;
}

double WeibCumHazardInv(double a, double rate, double shape)
{
  return pow(a / rate, 1.0 / shape);
}

// [[Rcpp::export]]

Rcpp::List WeibAugment(const arma::vec& failure_times,
                       const arma::mat& X,
                       const arma::vec& rate,
                       double shape
                       )
{
  int N = failure_times.size();
  int P = X.n_cols;

  std::vector<double> rejected_times; rejected_times.resize(0);
  std::vector<int> subject_id; subject_id.resize(0);

  // Augment the failure times
  for(int i = 0; i < N; i++) {
    double Lambda_0 = WeibCumHazard(failure_times[i], rate(i), shape);
    int num_reject = R::rpois(Lambda_0);
    for(int j = 0; j < num_reject; j++) {
      double A_bar = Lambda_0 * R::unif_rand();
      rejected_times.push_back(WeibCumHazardInv(A_bar, rate(i), shape));
      subject_id.push_back(i);
    }
  }

  int N_reject = rejected_times.size();
  mat X_reject = zeros<mat>(N_reject, P);
  for(int i = 0; i < N_reject; i++) {
    X_reject.row(i) = X.row(subject_id[i]);
  }

  List out;

  out["rejected_times"] = rejected_times;
  out["subject_id"] = subject_id;
  out["X_reject"] = X_reject;

  return out;

}

