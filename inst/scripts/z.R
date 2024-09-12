library(tidyverse)
library(spBayesSurv)
library(BART4RS)

data("leuk_data")

weibull_formula <-Surv(event_time, status) ~ age + wbc + tpi + sex

weibull_fit <-
  weibull_coxph(
    weibull_formula,
    data = leuk_data,
    pop_haz = rep(0, nrow(leuk_data)),
    test_data = leuk_data,
    verbose = FALSE
  )

## Get Coefficients and stuff --------------------------------------------------

weibull_samples <- as.matrix(weibull_fit)
shape_parameter <- mean(weibull_samples[,"weibull_shape"]) ## 0.574
scale_parameter <- mean(weibull_samples[,"weibull_scale"]) ## 4.82
beta_weibull    <- colMeans(weibull_samples[,grep("beta", colnames(weibull_samples))])

X <- model.matrix(~ age + wbc + tpi + sex - 1,
                  data = leuk_data)

linear_predictor <- X %*% beta_weibull %>% as.numeric()

## We want to simulate from shape / scale * (t / scale)^(shape - 1) * exp(linear_predictor)
## shape * exp(linear_predictor) / (scale^(shape)) * t^(shape - 1)
## shape * (exp(linear_predictor / shape) / scale)^shape * t^(shape - 1)

C         <- 14 * runif(nrow(X))
t_disease <- rweibull(n = nrow(X),
                      shape = shape_parameter,
                      scale = scale_parameter /
                        exp(linear_predictor / shape_parameter))
t_pop     <- rexp(n = nrow(X), rate = leuk_data$haz_rate)
t_both    <- pmin(t_disease, t_pop)

delta     <- ifelse(t_both < C, 1, 0)
Y         <- pmin(t_both, C)
leuk_data <- mutate(leuk_data, Y = Y, delta = delta)

fit_weib <- weibull_coxph(Surv(Y, delta) ~ age + wbc + tpi + sex,
                          pop_haz = leuk_data$haz_rate,
                          data = leuk_data)
weibull_samples2 <- as.matrix(fit_weib)
beta_weibull2    <- colMeans(weibull_samples2[,grep("beta", colnames(weibull_samples2))])

eta_hats <- X %*% beta_weibull2 %>% as.numeric()

plot(beta_weibull, beta_weibull2)
abline(a=0,b=1)

## Cox PH ----

fitted_cox <- coxpe_bart(Surv(Y, delta) ~ age + sex + wbc + tpi,
                         data = leuk_data, 
                         pop_haz = leuk_data$haz_rate)

plot(colMeans(fitted_cox$lambda_train), linear_predictor)

rmse <- function(x,y) sqrt(mean((x - y)^2))

cor(colMeans(fitted_cox$lambda_train), linear_predictor)
cor(eta_hats, linear_predictor)

rmse(eta_hats, linear_predictor)
rmse(colMeans(fitted_cox$lambda_train) - mean(fitted_cox$lambda_train) , linear_predictor - mean(linear_predictor))
