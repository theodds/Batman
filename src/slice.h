#ifndef SLICE_H
#define SLICE_H

#include <RcppArmadillo.h>
#include "functions.h"

struct Loglik {
  virtual double L(double x) {return 0;}
};

struct ShapeLoglik : Loglik {
  double sum_x;
  double sum_exp_x;
  double num_obs;

  double L(double x) {

    double alpha = 1.0 / (x * x);
    double out = num_obs * alpha * std::log(alpha)
      - num_obs * R::lgammafn(alpha)
      + alpha * sum_x - alpha * sum_exp_x;
      /* - 3.0 / 2.0 * std::log(alpha); */
      /* - std::log(1 + x * x); */
      /* - 0.5*x*x; */
      /* - x; */
      return out;

  }

  ShapeLoglik(double sumx, double sumexpx, double numobs) :
    sum_x(sumx), sum_exp_x(sumexpx), num_obs(numobs) {;}

};

struct RhoLoglik : Loglik {
  double mean_log_s;
  double p;
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  double L(double rho) {

    double alpha = alpha_scale * rho / (1.0 - rho);
    double beta_loglik = (alpha_shape_1 - 1.0) * log(rho) +
      (alpha_shape_2 - 1.0) * log(1.0 - rho);
    double loglik = alpha * mean_log_s + Rf_lgammafn(alpha) -
      p * Rf_lgammafn(alpha / p) + beta_loglik;
    return loglik;
  }

  RhoLoglik(double mean_log_s_, double p_, double alpha_scale_,
            double alpha_shape_1_, double alpha_shape_2_) :
    mean_log_s(mean_log_s_), p(p_), alpha_scale(alpha_scale_),
    alpha_shape_1(alpha_shape_1_), alpha_shape_2(alpha_shape_2_) {;}

};

struct SigmaLoglik : Loglik {
  double n;
  double SSE;
  double scale_sigma;

  double L(double x) {
    double tau = 1.0 / (x * x);
    double s = x / scale_sigma;
    double loglik = 0.5 * n * std::log(tau) - tau * 0.5 * SSE
      - std::log(1.0 + s * s);
    return loglik;
  }

  SigmaLoglik(double nx, double SSEx, double scalex) :
    n(nx), SSE(SSEx), scale_sigma(scalex) {;}

};

struct ScaleLambdaLoglik : Loglik {
  double n;
  double sum_lambda;
  double sum_exp_lambda;
  double scale;

  double L(double sigma) {
    double alpha,  beta;
    scale_lambda_to_alpha_beta(alpha, beta, sigma);
    /* Rcpp::Rcout << sigma; */
    double out = n * alpha * log(beta)
      - n * R::lgammafn(alpha)
      + alpha * sum_lambda
      - beta * sum_exp_lambda
      + M_LN2 - LN_2_BY_PI
      // - log(1.0 + pow(sigma/scale, 2.0));
      - 0.5 * pow(sigma / scale, 2.0);
    /* Rcpp::Rcout << " " << out << std::endl; */
    return out;
  }

 ScaleLambdaLoglik(double n_, double sum_lambda_,
                   double sum_exp_lambda_, double scale_)
   : n(n_), sum_lambda(sum_lambda_), sum_exp_lambda(sum_exp_lambda_),
    scale(scale_) {;}

};

double slice_sampler(double x0, Loglik* g, double w, double lower, double upper);


#endif
