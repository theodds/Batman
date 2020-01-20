x <- eta_1_1$estimate
tau <- eta_1_1$standard_error^2

f <- function(log_sigma_sq) {
  var_x <- exp(log_sigma_sq) + tau
  sd_x <- sqrt(var_x)
  sum(dnorm(x = x, mean = mean(x), sd = sd_x, log = TRUE))
}

plot(Vectorize(f), xlim = c(-10,10))

optimize(Vectorize(f), interval = c(-10, -5), maximum = TRUE)

x_s <- x - mean(x)

sqrt(sum(x_s^2) / length(x) - mean(tau))
