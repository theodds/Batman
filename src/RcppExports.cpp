// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Batman
Rcpp::List Batman(const arma::mat& X, const arma::vec& Y, const arma::sp_mat& probs, int num_cat, int num_tree, double scale_lambda, double shape_lambda_0, double rate_lambda_0, double scale_kappa, double sigma_scale_log_tau, double shape_tau_0, double rate_tau_0, int num_burn, int num_thin, int num_save, const arma::mat& X_test, const arma::vec& Y_test);
RcppExport SEXP _Batman_Batman(SEXP XSEXP, SEXP YSEXP, SEXP probsSEXP, SEXP num_catSEXP, SEXP num_treeSEXP, SEXP scale_lambdaSEXP, SEXP shape_lambda_0SEXP, SEXP rate_lambda_0SEXP, SEXP scale_kappaSEXP, SEXP sigma_scale_log_tauSEXP, SEXP shape_tau_0SEXP, SEXP rate_tau_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP, SEXP X_testSEXP, SEXP Y_testSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_cat(num_catSEXP);
    Rcpp::traits::input_parameter< int >::type num_tree(num_treeSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type shape_lambda_0(shape_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_lambda_0(rate_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type scale_kappa(scale_kappaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_scale_log_tau(sigma_scale_log_tauSEXP);
    Rcpp::traits::input_parameter< double >::type shape_tau_0(shape_tau_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_tau_0(rate_tau_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y_test(Y_testSEXP);
    rcpp_result_gen = Rcpp::wrap(Batman(X, Y, probs, num_cat, num_tree, scale_lambda, shape_lambda_0, rate_lambda_0, scale_kappa, sigma_scale_log_tau, shape_tau_0, rate_tau_0, num_burn, num_thin, num_save, X_test, Y_test));
    return rcpp_result_gen;
END_RCPP
}
// CoxBart
List CoxBart(const arma::mat& X, const arma::vec& Y, const arma::uvec& delta, const arma::uvec& order, const arma::uvec& L, const arma::uvec& U, const arma::sp_mat& probs, const arma::mat& X_test, int num_trees, double scale_lambda, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_CoxBart(SEXP XSEXP, SEXP YSEXP, SEXP deltaSEXP, SEXP orderSEXP, SEXP LSEXP, SEXP USEXP, SEXP probsSEXP, SEXP X_testSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type order(orderSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type U(USEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(CoxBart(X, Y, delta, order, L, U, probs, X_test, num_trees, scale_lambda, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// convertListToVector
std::vector<std::vector<int>> convertListToVector(Rcpp::List list);
RcppExport SEXP _Batman_convertListToVector(SEXP listSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type list(listSEXP);
    rcpp_result_gen = Rcpp::wrap(convertListToVector(list));
    return rcpp_result_gen;
END_RCPP
}
// CoxPEBart
List CoxPEBart(const arma::mat& X, const arma::vec& Y, const arma::uvec& delta, Rcpp::List bin_to_obs_list, const arma::uvec& obs_to_bin, const arma::vec& time_grid, const arma::vec& bin_width, const arma::vec& base_haz_init, const arma::sp_mat& probs, const arma::mat& X_test, int num_trees, double scale_lambda, bool do_rel_surv, const arma::vec& pop_haz_, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_CoxPEBart(SEXP XSEXP, SEXP YSEXP, SEXP deltaSEXP, SEXP bin_to_obs_listSEXP, SEXP obs_to_binSEXP, SEXP time_gridSEXP, SEXP bin_widthSEXP, SEXP base_haz_initSEXP, SEXP probsSEXP, SEXP X_testSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP do_rel_survSEXP, SEXP pop_haz_SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type bin_to_obs_list(bin_to_obs_listSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type obs_to_bin(obs_to_binSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type time_grid(time_gridSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type bin_width(bin_widthSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type base_haz_init(base_haz_initSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< bool >::type do_rel_surv(do_rel_survSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type pop_haz_(pop_haz_SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(CoxPEBart(X, Y, delta, bin_to_obs_list, obs_to_bin, time_grid, bin_width, base_haz_init, probs, X_test, num_trees, scale_lambda, do_rel_surv, pop_haz_, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// MLogitBart
List MLogitBart(const arma::mat& X, const arma::uvec& Y, const arma::sp_mat& probs, int num_cat, int num_trees, double scale_lambda, double shape_lambda_0, double rate_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_MLogitBart(SEXP XSEXP, SEXP YSEXP, SEXP probsSEXP, SEXP num_catSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP shape_lambda_0SEXP, SEXP rate_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_cat(num_catSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type shape_lambda_0(shape_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_lambda_0(rate_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(MLogitBart(X, Y, probs, num_cat, num_trees, scale_lambda, shape_lambda_0, rate_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// PoisBart
List PoisBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_PoisBart(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(PoisBart(X, Y, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// QBinomBart
List QBinomBart(const arma::mat& X, const arma::vec& Y, const arma::vec& n, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_QBinomBart(SEXP XSEXP, SEXP YSEXP, SEXP nSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(QBinomBart(X, Y, n, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// QGammaBart
List QGammaBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_QGammaBart(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(QGammaBart(X, Y, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// QMultinomBart
List QMultinomBart(const arma::mat& X, const arma::mat& Y, const arma::vec& n, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_QMultinomBart(SEXP XSEXP, SEXP YSEXP, SEXP nSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(QMultinomBart(X, Y, n, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// QNBBart
List QNBBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_QNBBart(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(QNBBart(X, Y, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// QPoisBart
List QPoisBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_lambda, double scale_lambda_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_QPoisBart(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP scale_lambda_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda_0(scale_lambda_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(QPoisBart(X, Y, X_test, probs, num_trees, scale_lambda, scale_lambda_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// RVarBart
List RVarBart(const arma::mat& X, const arma::vec& Y, const arma::sp_mat& probs, double sigma_scale_log_tau, double shape_tau_0, double rate_tau_0, int num_trees, int num_burn, int num_thin, int num_save, bool update_scale_log_tau, bool update_s);
RcppExport SEXP _Batman_RVarBart(SEXP XSEXP, SEXP YSEXP, SEXP probsSEXP, SEXP sigma_scale_log_tauSEXP, SEXP shape_tau_0SEXP, SEXP rate_tau_0SEXP, SEXP num_treesSEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP, SEXP update_scale_log_tauSEXP, SEXP update_sSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_scale_log_tau(sigma_scale_log_tauSEXP);
    Rcpp::traits::input_parameter< double >::type shape_tau_0(shape_tau_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_tau_0(rate_tau_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    Rcpp::traits::input_parameter< bool >::type update_scale_log_tau(update_scale_log_tauSEXP);
    Rcpp::traits::input_parameter< bool >::type update_s(update_sSEXP);
    rcpp_result_gen = Rcpp::wrap(RVarBart(X, Y, probs, sigma_scale_log_tau, shape_tau_0, rate_tau_0, num_trees, num_burn, num_thin, num_save, update_scale_log_tau, update_s));
    return rcpp_result_gen;
END_RCPP
}
// RegBart
List RegBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test, const arma::sp_mat& probs, int num_trees, double scale_sigma, double scale_sigma_mu, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_RegBart(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_sigmaSEXP, SEXP scale_sigma_muSEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_sigma(scale_sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type scale_sigma_mu(scale_sigma_muSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(RegBart(X, Y, X_test, probs, num_trees, scale_sigma, scale_sigma_mu, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// doit
void doit();
RcppExport SEXP _Batman_doit() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    doit();
    return R_NilValue;
END_RCPP
}
// WeibAugment
Rcpp::List WeibAugment(const arma::vec& failure_times, const arma::mat& X, const arma::vec& rate, double shape);
RcppExport SEXP _Batman_WeibAugment(SEXP failure_timesSEXP, SEXP XSEXP, SEXP rateSEXP, SEXP shapeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type failure_times(failure_timesSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type rate(rateSEXP);
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    rcpp_result_gen = Rcpp::wrap(WeibAugment(failure_times, X, rate, shape));
    return rcpp_result_gen;
END_RCPP
}
// VarBart
List VarBart(const arma::mat& X, const arma::vec& Y, const arma::sp_mat& probs, double scale_kappa, double sigma_scale_log_tau, double shape_tau_0, double rate_tau_0, int num_trees, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_VarBart(SEXP XSEXP, SEXP YSEXP, SEXP probsSEXP, SEXP scale_kappaSEXP, SEXP sigma_scale_log_tauSEXP, SEXP shape_tau_0SEXP, SEXP rate_tau_0SEXP, SEXP num_treesSEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< double >::type scale_kappa(scale_kappaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_scale_log_tau(sigma_scale_log_tauSEXP);
    Rcpp::traits::input_parameter< double >::type shape_tau_0(shape_tau_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_tau_0(rate_tau_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(VarBart(X, Y, probs, scale_kappa, sigma_scale_log_tau, shape_tau_0, rate_tau_0, num_trees, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// VarLogitBart
List VarLogitBart(const arma::mat& X_logit, const arma::uvec& Y_logit, const arma::mat& X_var, const arma::vec& Y_var, const arma::sp_mat& probs, int num_cat, int num_trees, double scale_lambda, double shape_lambda_0, double rate_lambda_0, double scale_kappa, double sigma_scale_log_tau, double shape_tau_0, double rate_tau_0, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_VarLogitBart(SEXP X_logitSEXP, SEXP Y_logitSEXP, SEXP X_varSEXP, SEXP Y_varSEXP, SEXP probsSEXP, SEXP num_catSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP shape_lambda_0SEXP, SEXP rate_lambda_0SEXP, SEXP scale_kappaSEXP, SEXP sigma_scale_log_tauSEXP, SEXP shape_tau_0SEXP, SEXP rate_tau_0SEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X_logit(X_logitSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y_logit(Y_logitSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_var(X_varSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y_var(Y_varSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_cat(num_catSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type shape_lambda_0(shape_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_lambda_0(rate_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type scale_kappa(scale_kappaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_scale_log_tau(sigma_scale_log_tauSEXP);
    Rcpp::traits::input_parameter< double >::type shape_tau_0(shape_tau_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_tau_0(rate_tau_0SEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(VarLogitBart(X_logit, Y_logit, X_var, Y_var, probs, num_cat, num_trees, scale_lambda, shape_lambda_0, rate_lambda_0, scale_kappa, sigma_scale_log_tau, shape_tau_0, rate_tau_0, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// WeibBart
List WeibBart(const arma::mat& X, const arma::vec& Y, const arma::vec& W, const arma::uvec& idx, const arma::sp_mat& probs, int num_trees, double scale_lambda, double shape_lambda_0, double rate_lambda_0, double weibull_power, bool do_ard, bool update_alpha, bool update_scale, int num_burn, int num_thin, int num_save);
RcppExport SEXP _Batman_WeibBart(SEXP XSEXP, SEXP YSEXP, SEXP WSEXP, SEXP idxSEXP, SEXP probsSEXP, SEXP num_treesSEXP, SEXP scale_lambdaSEXP, SEXP shape_lambda_0SEXP, SEXP rate_lambda_0SEXP, SEXP weibull_powerSEXP, SEXP do_ardSEXP, SEXP update_alphaSEXP, SEXP update_scaleSEXP, SEXP num_burnSEXP, SEXP num_thinSEXP, SEXP num_saveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type idx(idxSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< double >::type scale_lambda(scale_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type shape_lambda_0(shape_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type rate_lambda_0(rate_lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type weibull_power(weibull_powerSEXP);
    Rcpp::traits::input_parameter< bool >::type do_ard(do_ardSEXP);
    Rcpp::traits::input_parameter< bool >::type update_alpha(update_alphaSEXP);
    Rcpp::traits::input_parameter< bool >::type update_scale(update_scaleSEXP);
    Rcpp::traits::input_parameter< int >::type num_burn(num_burnSEXP);
    Rcpp::traits::input_parameter< int >::type num_thin(num_thinSEXP);
    Rcpp::traits::input_parameter< int >::type num_save(num_saveSEXP);
    rcpp_result_gen = Rcpp::wrap(WeibBart(X, Y, W, idx, probs, num_trees, scale_lambda, shape_lambda_0, rate_lambda_0, weibull_power, do_ard, update_alpha, update_scale, num_burn, num_thin, num_save));
    return rcpp_result_gen;
END_RCPP
}
// rlgam
double rlgam(double shape);
RcppExport SEXP _Batman_rlgam(SEXP shapeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    rcpp_result_gen = Rcpp::wrap(rlgam(shape));
    return rcpp_result_gen;
END_RCPP
}
// gaussian_gaussian_marginal_loglik
double gaussian_gaussian_marginal_loglik(double n, double sum_y, double sum_y_sq, double prec_y, double prec_mu);
RcppExport SEXP _Batman_gaussian_gaussian_marginal_loglik(SEXP nSEXP, SEXP sum_ySEXP, SEXP sum_y_sqSEXP, SEXP prec_ySEXP, SEXP prec_muSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type sum_y(sum_ySEXP);
    Rcpp::traits::input_parameter< double >::type sum_y_sq(sum_y_sqSEXP);
    Rcpp::traits::input_parameter< double >::type prec_y(prec_ySEXP);
    Rcpp::traits::input_parameter< double >::type prec_mu(prec_muSEXP);
    rcpp_result_gen = Rcpp::wrap(gaussian_gaussian_marginal_loglik(n, sum_y, sum_y_sq, prec_y, prec_mu));
    return rcpp_result_gen;
END_RCPP
}
// trigamma_inverse
double trigamma_inverse(double x);
RcppExport SEXP _Batman_trigamma_inverse(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(trigamma_inverse(x));
    return rcpp_result_gen;
END_RCPP
}
// gamma_gamma_shape_update
double gamma_gamma_shape_update(const arma::vec& tau, double mu, double alpha, double beta);
RcppExport SEXP _Batman_gamma_gamma_shape_update(SEXP tauSEXP, SEXP muSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_gamma_shape_update(tau, mu, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _Batman_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _Batman_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _Batman_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _Batman_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_var_forest();
RcppExport SEXP _rcpp_module_boot_weib_forest();

static const R_CallMethodDef CallEntries[] = {
    {"_Batman_Batman", (DL_FUNC) &_Batman_Batman, 17},
    {"_Batman_CoxBart", (DL_FUNC) &_Batman_CoxBart, 13},
    {"_Batman_convertListToVector", (DL_FUNC) &_Batman_convertListToVector, 1},
    {"_Batman_CoxPEBart", (DL_FUNC) &_Batman_CoxPEBart, 17},
    {"_Batman_MLogitBart", (DL_FUNC) &_Batman_MLogitBart, 11},
    {"_Batman_PoisBart", (DL_FUNC) &_Batman_PoisBart, 10},
    {"_Batman_QBinomBart", (DL_FUNC) &_Batman_QBinomBart, 11},
    {"_Batman_QGammaBart", (DL_FUNC) &_Batman_QGammaBart, 10},
    {"_Batman_QMultinomBart", (DL_FUNC) &_Batman_QMultinomBart, 11},
    {"_Batman_QNBBart", (DL_FUNC) &_Batman_QNBBart, 10},
    {"_Batman_QPoisBart", (DL_FUNC) &_Batman_QPoisBart, 10},
    {"_Batman_RVarBart", (DL_FUNC) &_Batman_RVarBart, 12},
    {"_Batman_RegBart", (DL_FUNC) &_Batman_RegBart, 10},
    {"_Batman_doit", (DL_FUNC) &_Batman_doit, 0},
    {"_Batman_WeibAugment", (DL_FUNC) &_Batman_WeibAugment, 4},
    {"_Batman_VarBart", (DL_FUNC) &_Batman_VarBart, 11},
    {"_Batman_VarLogitBart", (DL_FUNC) &_Batman_VarLogitBart, 17},
    {"_Batman_WeibBart", (DL_FUNC) &_Batman_WeibBart, 16},
    {"_Batman_rlgam", (DL_FUNC) &_Batman_rlgam, 1},
    {"_Batman_gaussian_gaussian_marginal_loglik", (DL_FUNC) &_Batman_gaussian_gaussian_marginal_loglik, 5},
    {"_Batman_trigamma_inverse", (DL_FUNC) &_Batman_trigamma_inverse, 1},
    {"_Batman_gamma_gamma_shape_update", (DL_FUNC) &_Batman_gamma_gamma_shape_update, 4},
    {"_Batman_rcpparma_hello_world", (DL_FUNC) &_Batman_rcpparma_hello_world, 0},
    {"_Batman_rcpparma_outerproduct", (DL_FUNC) &_Batman_rcpparma_outerproduct, 1},
    {"_Batman_rcpparma_innerproduct", (DL_FUNC) &_Batman_rcpparma_innerproduct, 1},
    {"_Batman_rcpparma_bothproducts", (DL_FUNC) &_Batman_rcpparma_bothproducts, 1},
    {"_rcpp_module_boot_var_forest", (DL_FUNC) &_rcpp_module_boot_var_forest, 0},
    {"_rcpp_module_boot_weib_forest", (DL_FUNC) &_rcpp_module_boot_weib_forest, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_Batman(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
