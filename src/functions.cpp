#include "functions.h"

#include <RcppArmadillo.h>

#ifndef M_SQRT_2PI
#define M_SQRT_2PI 2.5066282746310005024157652848
#endif

int sample_class(const arma::vec& probs) {
  double U = R::unif_rand();
  double foo = 0.0;
  int K = probs.size();

  // Sample
  for(int k = 0; k < K; k++) {
    foo += probs(k);
    if(U < foo) {
      return(k);
    }
  }
  return K - 1;
}

int sample_class(int n) {
  double U = R::unif_rand();
  double p = 1.0 / ((double)n);
  double foo = 0.0;

  for(int k = 0; k < n; k++) {
    foo += p;
    if(U < foo) {
      return k;
    }
  }
  return n - 1;
}

int sample_class_col(const arma::sp_mat& probs, int col) {
  double U = R::unif_rand();
  double cumsum = 0.0;

  arma::sp_mat::const_col_iterator it = probs.begin_col(col);
  arma::sp_mat::const_col_iterator it_end = probs.end_col(col);
  for(; it != it_end; ++it) {
    cumsum += (*it);
    if(U < cumsum) {
      return it.row();
    }
  }
  return it.row();

}

double logit(double x) {
  return log(x) - log(1.0-x);
}

double expit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double log_sum_exp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}

double log_sum_exp(double a, double b) {
  double M = (a < b ? b : a);
  return M + log(exp(a - M) + exp(b - M));
}

// [[Rcpp::export]]
double rlgam(double shape) {
  if(shape >= 0.1) return log(Rf_rgamma(shape, 1.0));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = unif_rand();
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(unif_rand()) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(unif_rand())) break;
  } while(true);

  // Rcout << "Sample: " << -z/a << "\n";

  return -z/a;
}

arma::vec rdirichlet(const arma::vec& shape) {
  arma::vec out = arma::zeros<arma::vec>(shape.size());
  for(int i = 0; i < shape.size(); i++) {
    do {
      out(i) = Rf_rgamma(shape(i), 1.0);
    } while(out(i) == 0);
  }
  out = out / arma::sum(out);
  return out;
}

double alpha_to_rho(double alpha, double scale) {
  return alpha / (alpha + scale);
}

double rho_to_alpha(double rho, double scale) {
  return scale * rho / (1.0 - rho);
}

double logpdf_beta(double x, double a, double b) {
  return (a-1.0) * log(x) + (b-1.0) * log(1 - x) - Rf_lbeta(a,b);
}

bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new) {

  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;

  return log(unif_rand()) < cutoff ? true : false;

}

double randnt(double lower, double upper) {
  if( (lower <= 0 && upper == INFINITY) ||
      (upper >= 0 && lower == -INFINITY) ||
      (lower <= 0 && upper >= 0 && upper - lower > M_SQRT_2PI)) {
    while(true) {
      double r = R::norm_rand();
      if(r > lower && r < upper)
	return(r);
    }
  }
  else if( (lower > 0) &&
	   (upper - lower >
	    2.0 / (lower + sqrt(lower * lower + 4.0)) *
	    exp((lower * lower - lower * sqrt(lower * lower + 4.0)) / 4.0))
	   )
    {
      double a = (lower + sqrt(lower * lower + 4.0))/2.0;
      while(true) {
	double r = R::exp_rand() / a + lower;
	double u = R::unif_rand();
	if (u < exp(-0.5 * pow(r - a, 2)) && r < upper) {
	  return(r);
	}
      }
    }
  else if ( (upper < 0) &&
	    (upper - lower >
	     2.0 / (-upper + sqrt(upper * upper + 4.0)) *
	     exp((upper*upper + upper * sqrt(upper * upper + 4.0)) / 4.0))
	    )
    {
      double a = (-upper + sqrt(upper*upper + 4.0)) / 2.0;
      while(true) {
	double r = R::exp_rand() / a - upper;
	double u = R::unif_rand();
	if (u < exp(-0.5 * pow(r - a, 2)) && r < -lower) {
	  return(-r);
	}
      }
    }
  else {
    while(true) {
      double r = lower + R::unif_rand() * (upper - lower);
      double u = R::unif_rand();
      double rho;
      if (lower > 0) {
	rho = exp((lower*lower - r*r) * 0.5);
      }
      else if (upper < 0) {
	rho = exp((upper*upper - r*r) * 0.5);
      }
      else {
	rho = exp(-r*r * 0.5);
      }
      if(u < rho) {
	return(r);
      }
    }
  }
}

double randnt(double mu, double sigma, double lower, double upper) {
  double Z = randnt( (lower - mu) / sigma,
		     (upper - mu) / sigma
		     );
  return(mu + sigma * Z);
}

double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = pow(tau, -0.5);
  int give_log = 1;

  double out = Rf_dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);

  return out;

}

// TODO: Test this for correctness
double half_cauchy_update_precision_mh(const arma::vec& R,
                                       double prec_old,
                                       double sigma_scale) {
  double SSE = sum(R % R);
  double n = R.size();
  double shape = 0.5 *n + 1;
  double scale = 2.0 / SSE;
  double tau_prop = Rf_rgamma(shape, scale);
  double sigma_prop = pow(tau_prop, -0.5);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_scale) -
    cauchy_jacobian(prec_old, sigma_scale);

  return log(unif_rand()) < loglik_rat ? tau_prop : prec_old;

}

// [[Rcpp::export]]
double gaussian_gaussian_marginal_loglik(double n,
                                         double sum_y,
                                         double sum_y_sq,
                                         double prec_y,
                                         double prec_mu) {

  double y_bar = (n == 0.0 ? 0.0 : sum_y / n);
  double SSE = (n == 0.0 ? 0.0 : sum_y_sq - n * y_bar * y_bar);

  double out = 0.5 * n * log(prec_y) - 0.5 * n * LN_2PI;
  out += 0.5 * log(prec_mu) - 0.5 * log(n * prec_y + prec_mu);
  out += -0.5 * prec_y * SSE;
  out += -0.5 * y_bar * y_bar * n * prec_y * prec_mu / (n * prec_y + prec_mu);

  // Rcpp::Rcout << out << " " << n << " " << sum_y << " " <<
  //   sum_y_sq << " " << prec_y << " " << prec_mu << std::endl;


  return out;

}

double gaussian_gaussian_draw_posterior(double n,
                                        double sum_y,
                                        double prec_y,
                                        double prec_mu) {

  double mu_up = prec_y * sum_y / (n * prec_y + prec_mu);
  double sigma_up = pow(n * prec_y + prec_mu, -0.5);
  return mu_up + sigma_up * norm_rand();
}

double weighted_normal_mean0_gamma_loglik(double n,
                                          double sum_eta,
                                          double sum_eta_y,
                                          double sum_eta_y_sq,
                                          double sum_log_eta,
                                          double alpha,
                                          double beta) {

  double out = 0.;
  if(n == 0) return out;

  double alpha_up = alpha + 0.5 * n;
  double beta_up = beta + 0.5 * sum_eta_y_sq;
  
  out += alpha * log(beta) - alpha_up * log(beta_up);
  out += R::lgammafn(alpha_up) - R::lgammafn(alpha);
  out += 0.5 * sum_log_eta - 0.5 * n * LN_2PI;

  return out;
}

double weighted_normal_mean0_gamma_draw(double n,
                                        double sum_eta,
                                        double sum_eta_y,
                                        double sum_eta_y_sq,
                                        double sum_log_eta,
                                        double alpha,
                                        double beta) {
  double out = 0.;
  if(n == 0) return out;

  double alpha_up = alpha + 0.5 * n;
  double beta_up = beta + 0.5 * sum_eta_y_sq;

  return R::rgamma(alpha_up, 1.0 / beta_up);
}

double weighted_normal_gamma_loglik(int n,
                                    double sum_eta,
                                    double sum_eta_y,
                                    double sum_eta_y_sq,
                                    double sum_log_eta,
                                    double alpha, double beta, double kappa)
{
  double out = 0.;
  if(n == 0) return out;

  double y_tilde = sum_eta_y / sum_eta;
  double SSE     = sum_eta_y_sq - sum_eta * y_tilde * y_tilde;

  double alpha_up = alpha + 0.5 * n;
  double beta_up = beta + 0.5 * SSE
    + 0.5 * pow(y_tilde, 2) * kappa * sum_eta / (kappa + sum_eta);
  double kappa_up = kappa + sum_eta;
  double mu_up = sum_eta * y_tilde / (sum_eta + kappa);

  out += alpha * log(beta) - alpha_up * log(beta_up);
  out += R::lgammafn(alpha_up) - R::lgammafn(alpha);
  out += 0.5 * sum_log_eta - 0.5 * n * LN_2PI;
  out += 0.5 * log(kappa) - 0.5 * log(kappa_up);

  return out;
}

arma::vec weighted_normal_gamma_draw_posterior(int n,
                                               double sum_eta,
                                               double sum_eta_y,
                                               double sum_eta_y_sq,
                                               double sum_log_eta,
                                               double alpha, double beta, double kappa)
{

  arma::vec mu_tau = arma::zeros<arma::vec>(2);

  if(n == 0) {
    mu_tau(1) = R::rgamma(alpha, 1.0 / beta);
    mu_tau(0) = norm_rand() / sqrt(mu_tau(1) * kappa);

    return mu_tau;
  }

  double y_tilde = (n == 0 ? 0.0 : sum_eta_y / sum_eta);
  double SSE = sum_eta_y_sq - sum_eta * y_tilde * y_tilde;

  double alpha_up = alpha + 0.5 * n;
  double beta_up = beta + 0.5 * SSE
    + 0.5 * pow(y_tilde, 2) * kappa * sum_eta / (kappa + sum_eta);
  double kappa_up = kappa + sum_eta;
  double mu_up = sum_eta * y_tilde / (sum_eta + kappa);

  double tau = R::rgamma(alpha_up, 1.0 / beta_up);
  double mu = mu_up + pow(tau * kappa_up, -0.5) * norm_rand();

  mu_tau(0) = mu;
  mu_tau(1) = tau;
  return mu_tau;

}

double poisson_lgamma_marginal_loglik(double sum_y,
                                      double sum_y_lambda_minus,
                                      double sum_exp_lambda_minus,
                                      double alpha, double beta)
{
  double alpha_up = alpha + sum_y;
  double beta_up = beta + sum_exp_lambda_minus;
  double out = alpha * log(beta) - R::lgammafn(alpha);
  out += R::lgammafn(alpha_up) - alpha_up * log(beta_up);
  return out;
}

double poisson_lgamma_draw_posterior(double sum_y,
                                     double sum_y_lambda_minus,
                                     double sum_exp_lambda_minus,
                                     double alpha, double beta)
{
  double alpha_up = alpha + sum_y;
  double beta_up  = beta + sum_exp_lambda_minus;
  return rlgam(alpha_up) - log(beta_up);
}

void scale_lambda_to_alpha_beta(double& alpha, double& beta, const double scale_lambda) {
  double scale_sq = scale_lambda * scale_lambda;
  // alpha = (1.0 + pow(1. + 2. * scale_sq, 0.5)) / (2. * scale_sq);
  alpha = trigamma_inverse(scale_sq);
  beta = exp(R::digamma(alpha));
}


// Code taken from the limma package in R.
// Algorithm originally written by Gordon Smyth (9/8/2002).

// [[Rcpp::export]]
double trigamma_inverse(double x) {

  // Very large and very small values - deal with using asymptotics
  if(x > 1E7) {
    return 1. / sqrt(x);
  }
  if(x < 1E-6) {
    return 1 / x;
  }

  // Otherwise, use Newton's method.
  double y = 0.5 + 1.0/x;
  for(int i = 0; i < 50; i++) {
    double tri = R::trigamma(y);
    double dif = tri * (1 - tri / x) / R::tetragamma(y);
    y += dif;
    if(-dif / y < 1E-8) break;
  }

  return y;

}

arma::uvec update_gmm_class(const arma::vec& obs,
                            const arma::vec& loc_obs,
                            const arma::vec& sigma_obs,
                            const arma::vec& loc_class,
                            const arma::vec& sigma_class,
                            const arma::mat& log_class_weights) {

  static const int give_log = 1;

  int N = obs.size();
  int K = loc_class.size();
  arma::uvec Z = arma::zeros<arma::uvec>(N);
  // mat log_weights = log_class_weights;
  for(int n = 0; n < N; n++) {
    arma::vec log_weights = trans(log_class_weights.row(n));
    for(int k = 0; k < K; k++) {
      double mu = loc_obs(n) + loc_class(k);
      double sigma = sigma_obs(n) * sigma_class(k);
      log_weights(k) = log_weights(k) + R::dnorm4(obs(n), mu, sigma, give_log);
    }
    arma::vec probs = exp(log_weights - max(log_weights));
    probs = probs / arma::sum(probs);
    Z(n) = sample_class(probs);
  }

  return Z;

}

// [[Rcpp::export]]
double gamma_gamma_shape_update(const arma::vec& tau,
                                double mu,
                                double alpha, double beta)
{

  static const int M = 10;
  static const double epsilon = 1E-8;

  int n = tau.size();
  double R = sum(log(tau));
  double S = sum(tau);
  double T = S / mu - R + n * log(mu) - n;
  double A = alpha + 0.5 * n;
  double B = beta + T;

  for(int j = 0; j < M; j++) {
    double a = A / B;
    A = alpha - n * a + n * a * a * R::trigamma(a);
    B = beta + (A - alpha) / a - n * log(a) + n * R::digamma(a) + T;
    double error = abs(a / (A / B) - 1.0);
    if(error < epsilon) break;
  }

  return R::rgamma(A, 1.0/B);

}

