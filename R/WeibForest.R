MakeWeib <- function(probs,
                     num_trees,
                     scale_lambda,
                     shape_lambda_0,
                     rate_lambda_0,
                     weibull_power,
                     update_scale) {

  mf <- Module(module = "weib_forest", PACKAGE = "Batman")
  return(new(mf$WeibModel,
             probs,
             num_trees,
             scale_lambda,
             shape_lambda_0,
             rate_lambda_0,
              weibull_power,
             update_scale))
}
