MakeRVar <- function(probs,
                     num_tree = 50,
                     shape_tau_0 = 0.1,
                     rate_tau_0 = 0.1,
                     k = 1.5
                     ) {

  if(is.null(sigma_scale_log_tau)) sigma_scale_log_tau <- k / sqrt(num_tree)
  
  hypers <- list(probs = probs,
                 sigma_scale_log_tau = sigma_scale_log_tau,
                 shape_tau_0 = shape_tau_0,
                 rate_tau_0 = rate_tau_0,
                 num_tree = num_tree)
  opts <- list()
  
  mf <- Module(module = "var_forest", PACKAGE = "Batman")
  return(new(mf$RVarForest, hypers, opts))
}
