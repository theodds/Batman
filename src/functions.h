#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <RcppArmadillo.h>

static double LN_2PI = 1.83787706640934548356065947281;
static double LN_2_BY_PI = -0.4515827052894548647261952;

int sample_class(const arma::vec& probs);

int sample_class(int n);

int sample_class_col(const arma::sp_mat& probs, int col);

double logit(double x);

double expit(double x);

double log_sum_exp(const arma::vec& x);

double log_sum_exp(double a, double b);

double rlgam(double shape);

arma::vec rdirichlet(const arma::vec& shape);

double alpha_to_rho(double alpha, double scale);

double rho_to_alpha(double rho, double scale);

double logpdf_beta(double x, double a, double b);

bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new);

double randnt(double lower, double upper);

double randnt(double mu, double sigma, double lower, double upper);

double cauchy_jacobian(double tau, double sigma_hat);

double half_cauchy_update_precision_mh(const arma::vec& R,
                                       double prec_old,
                                       double sigma_scale);

double gaussian_gaussian_marginal_loglik(double n,
                                         double sum_y,
                                         double sum_y_sq,
                                         double prec_y,
                                         double prec_mu);

double gaussian_gaussian_draw_posterior(double n,
                                        double sum_y,
                                        double prec_y,
                                        double prec_mu);

double weighted_normal_mean0_gamma_loglik(double n,
                                          double sum_eta,
                                          double sum_eta_y,
                                          double sum_eta_y_sq,
                                          double sum_log_eta,
                                          double alpha,
                                          double beta);

double weighted_normal_gamma_loglik(int n,
                                    double sum_eta,
                                    double sum_eta_y,
                                    double sum_eta_y_sq,
                                    double sum_log_eta,
                                    double alpha, double beta, double kappa);

arma::vec weighted_normal_gamma_draw_posterior(int n,
                                               double sum_eta,
                                               double sum_eta_y,
                                               double sum_eta_y_sq,
                                               double sum_log_eta,
                                               double alpha, double beta, double kappa);

double poisson_lgamma_marginal_loglik(double sum_y,
                                      double sum_y_lambda_minus,
                                      double sum_exp_lambda_minus,
                                      double alpha, double beta);

double poisson_lgamma_draw_posterior(double sum_y,
                                     double sum_y_lambda_minus,
                                     double sum_exp_lambda_minus,
                                     double alpha, double beta);

double trigamma_inverse(double x);

void scale_lambda_to_alpha_beta(double& alpha, double& beta, const double scale_lambda);

arma::uvec update_gmm_class(const arma::vec& obs,
                            const arma::vec& loc_obs,
                            const arma::vec& sigma_obs,
                            const arma::vec& loc_class,
                            const arma::vec& sigma_class,
                            const arma::mat& log_class_weights);

double gamma_gamma_shape_update(const arma::vec& tau, double mu,
                                double alpha, double beta);

#endif
