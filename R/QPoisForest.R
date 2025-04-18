MakeQPois <- function(probs,
                      num_tree = 50,
                      k = 1.5,
                      update_s = TRUE,
                      phi = 1
                      ) {

  sigma_scale_lambda <- k / sqrt(num_tree)
  
  hypers <- list(probs = probs, 
                 sigma_scale_lambda = sigma_scale_lambda, 
                 scale_lambda = sigma_scale_lambda,
                 phi = phi,
                 num_tree = num_tree,
                 update_s = TRUE
                 )
  opts <- list()

  mf <- Rcpp::Module(module = "qpois_forest", PACKAGE = "Batman")
  return(new(mf$QPoisForest, hypers, opts))

}


