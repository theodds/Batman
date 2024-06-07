# Packages required ---
# require(brms)
# require(loo)
require(tidyverse)
require(SoftBart)
install.packages("devtools")
devtools::install_github("spencerwoody/possum")
library(possum)
library(mgcv)
library(stringr)
library(ggplot2)
library(latex2exp)
library(patchwork)
# meps2020 <- readRDS("~/Documents/Works/Tony/data/meps2020.rds")

# PH ---------------------------------------------------------------------------
meps <- readRDS("~/Documents/Works/Tony/data/meps.rds")

formula <- phealth ~ -1 + age + bmi + edu + income + povlev + region + sex + marital + race + seatbelt + smoke
set.seed(123)
trnIndx <- sample(1:nrow(meps), ceiling(0.75 * nrow(meps)), replace = F) 
X <- (model.frame(formula, data = meps) %>% 
  select(-phealth) %>% preprocess_df())[[1]] %>% quantile_normalize_bart()
Y <-  meps$phealth
trnX <- X[trnIndx, ]
trnY <- Y[trnIndx]
tstX <- X[-trnIndx, ]
tstY <- Y[-trnIndx]

n_burn <- 2500
n_save <- 2500

ph <- CLogLogOrdinalBart(
  X = trnX,
  Y = trnY - 1,
  num_levels = 5,
  X_test = tstX,
  probs = Matrix::Matrix(diag(ncol(trnX))),
  num_trees = 50,
  scale_lambda = 2 / sqrt(50),
  alpha_gamma = 2,
  beta_gamma = 2,
  gamma_0 = log(-log(mean(trnY > 1))),
  num_burn = n_burn,
  num_thin = 1,
  num_save = n_save
)
# Notes. Using gamma_0 restriction 
saveRDS(ph, file = "ph.rds")

Y_max <- max(trnY)
Y_min <- min(trnY)

# prediction accuracy

logprPH <- function(y, gamma, lambda){
  ifelse(y == Y_min, 
         0,
         - sum(exp(gamma[1:(y-1)] + lambda)))  + 
    
    ifelse(y == Y_max,
           0, 
           log(1 - exp(- exp(gamma[y] + lambda))))
}

prPH <- matrix(0, nrow = nrow(tstX), ncol = n_save)
for(i in 1:nrow(tstX)){
  for(itr in 1:n_save){
    prPH[i, itr] <- logprPH(tstY[i], ph$gamma[itr, ],
                            ph$lambda_test[itr, i])
  }
}

pred_probPH <- log(rowSums(exp(prPH)) / n_save)  # log predictive prob for test data
pred_probPH <- log(apply(exp(prPH), 1, median))
hist(pred_probPH)
hist(exp(pred_probPH))

colMeans(ph$gamma)

# NPH --------------------------------------------------------------------------
trnXwj <- cbind(trnX, 0)
tstXwj <- cbind(tstX, 0)
sVec <- c(diag(ncol(trnXwj))) 
sVec[length(sVec)] <- 0.5
s <- Matrix::Matrix(sVec, ncol(trnXwj), ncol(trnXwj))
bin_to_list <- lapply(1:Y_max, function(i) which(trnY == i) - 1)

n_burn <- 2500
n_save <- 2500

nph <- CoxNPHOBart(trnXwj, 
                   trnY - 1, 
                   bin_to_list, 
                   probs = s, 
                   X_test = tstXwj, 
                   num_trees = 50, 
                   scale_lambda = 2 / sqrt(50), 
                   shape_gamma = 1, 
                   rate_gamma = 1, 
                   num_burn = n_burn, 
                   num_thin = 1, 
                   num_save = n_save
                   )

saveRDS(nph, file = "nph.rds")

Y_max <- max(trnY)
Y_min <- min(trnY)

# prediction accuracy
logprNPH <- function(y, gamma, lambda){
  ifelse(y == Y_min, 
         0,
         - sum(exp(gamma[1:(y-1)] + lambda[1:(y-1)]))) + 
    ifelse(y == Y_max,
           0, 
           log(1 - exp(- exp(gamma[y] + lambda[y]))))
}

prNPH <- matrix(0, nrow = nrow(tstX), ncol = n_save)
for(i in 1:nrow(tstX)){
  for(itr in 1:n_save){
    prNPH[i, itr] <- logprNPH(tstY[i], nph$gamma[itr, ],
                            nph$lambda_test[i, ,itr])
  }
}

pred_probNPH <- log(rowSums(exp(prNPH)) / n_save)  # log predictive prob for test data
pred_probNPH <- log(apply(exp(prNPH), 1, median))
hist(pred_probNPH)
hist(exp(pred_probNPH))

colMeans(nph$gamma)

dat <- data.frame(x = rep(1:nrow(tstX), 2), 
                  y = exp(c(pred_probPH, pred_probNPH)), 
                 grp = rep(c("PH", "NPH"), each = nrow(tstX)))
dat$grp <- factor(dat$grp, levels = c("PH", "NPH")) 

ggplot() +
  geom_density(data = dat, mapping = aes(x = y, fill = grp), alpha = 0.4) + 
  labs(x = "predictive prob") +
  theme_bw() +
  theme(legend.position = "bottom")
  
mean(pred_probNPH > pred_probPH)

dat <- data.frame(x = exp(pred_probPH), y = exp(pred_probNPH))

ggplot() +
  geom_point(data = dat, mapping = aes(x = x, y = y), col = "blue", size = 0.3) +
  geom_abline(col = "green", lwd = 0.8) +
  labs(x = "predictive prob for PH", y = "predictive prob for NPH") +
  theme_bw()

# posterior summarization
pos_summary <- function(lambda_samples, lambda_hat, formula, trnX){
  return(
    possum::additive_summary(
      formula,
      fhatSamples = lambda_samples,
      fhat = lambda_hat,
      df = trnX %>% as.data.frame()
  )
  )
}

# Formulas
formula <- lambda_hat ~ s(age) + s(bmi) + s(edu) + s(income) + s(povlev) + factor(sexMale) + factor(smoke)
formula_noage <- lambda_hat ~ s(bmi) + s(edu) + s(income) + s(povlev) + factor(sexMale) + factor(smoke)
formula_nosmoke <- lambda_hat ~ s(age) + s(bmi) + s(edu) + s(income) + s(povlev) + factor(sexMale)
formula_noedu <- lambda_hat ~ s(age) + s(bmi) + s(income) + s(povlev) + factor(sexMale) + factor(smoke)
formula_nopovlev <- lambda_hat ~ s(age) + s(bmi) + s(edu) + s(povlev) + factor(sexMale) + factor(smoke)
formula_nosex <- lambda_hat ~ s(age) + s(bmi) + s(edu) + s(income) + s(povlev) + factor(smoke)
formula_nobmi <- lambda_hat ~ s(age) + s(edu) + s(income) + s(povlev) + factor(sexMale) + factor(smoke)

# Plot fns
additive_summary_plot_2 <- function (additive_summary,
                                     ribbonFill = "grey80",
                                     windsor = NA)
  
{
  temp <- additive_summary$gamDf %>%
    mutate(term = case_when(
      TRUE ~ term
    ))
  if (!is.na(windsor)) {
    if (!("quant" %in% colnames(temp))) {
      stop("Quantiles not supplied")
    }
    temp <- temp %>% filter(quant > windsor/2 & quant < 1 -
                              
                              windsor/2)
    glimpse(temp)
  }
  temp %>% distinct() %>%
    ggplot() + geom_hline(yintercept = 0) +
    geom_ribbon(aes(x_j, ymin = fx_j_lo, ymax = fx_j_hi),
                fill = ribbonFill, alpha = 0.5) +
    geom_line(aes(x_j, fx_j_mean), col = "firebrick3") +
    geom_rug(aes(x_j, fx_j_mean), sides = "b", alpha = 0.25) +
    facet_wrap(~term, scale = "free") +
    labs(x = ("term"), y = ("Partial effect"))
}

rsq_df <- function(RsqVec, legendVec){
  return(data.frame(Rsq = RsqVec,
                         Model = rep(legendVec,
                                     each = 2500))
         )
}

plt_rsq <- function(df){
  return(ggplot(df) +
           geom_density(aes(x = Rsq, y = after_stat(density), fill = Model),
                        color = 'white', alpha = 0.3) +
           xlab(TeX("Summary $R^2$")) +
           ylab("Density") + 
           theme_bw())
}

# For PH
lambda_samplesPH <- ph$lambda
lambda_hat <- colMeans(lambda_samplesPH) %>% as.numeric()

possumPH <- pos_summary(t(lambda_samplesPH), lambda_hat, formula, trnX)
possumPH_noage <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_noage, trnX)  
possumPH_nosmoke <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_nosmoke, trnX)  
possumPH_noedu <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_noedu, trnX)
possumPH_noincome <- pos_summary(t(lambda_samplesPH), lambda_hat, formula, trnX)
possumPH_nopovlev <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_nopovlev, trnX)
possumPH_nosex <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_nosex, trnX)
possumPH_nobmi <- pos_summary(t(lambda_samplesPH), lambda_hat, formula_nobmi, trnX)

add_sumPH <- additive_summary_plot_2(possumPH) + xlab("") + theme_bw()

RsqVecPH <- c(possumPH$summaryRsq,
              possumPH_noage$summaryRsq,
              #possumPH_nobmi$summaryRsq,
              possumPH_noedu$summaryRsq,
              possumPH_noincome$summaryRsq
              #possumPH_nopovlev$summaryRsq
              #possumPH_nosex$summaryRsq,
              #possumPH_nosmoke$summaryRsq
              )
legendVecPH <- c("All", "Without age", #"Without bmi", 
                 "Without edu", "Without income" 
                 #,"Without povlev" #, "Without sex" #, "Without smoke"
               )

pltPH <- plt_rsq(rsq_df(RsqVecPH, legendVecPH))

# posterior summarization: NPH
lambdaNPH_samples <- nph$lambda_train


# For NPH category 1
lambda_hat <- rowMeans(lambdaNPH_samples[, 1, ]) %>% as.numeric()

possumNPH1 <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula, trnX)
possumNPH1_noage <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_noage, trnX)  
possumNPH1_nosmoke <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_nosmoke, trnX)  
possumNPH1_noedu <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_noedu, trnX)
possumNPH1_noincome <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula, trnX)
possumNPH1_nopovlev <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_nopovlev, trnX)
possumNPH1_nosex <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_nosex, trnX)
possumNPH1_nobmi <- pos_summary(lambdaNPH_samples[, 1, ], lambda_hat, formula_nobmi, trnX)

add_sumNPH1 <- additive_summary_plot_2(possumNPH1) + xlab("") + theme_bw()

RsqVecNPH1 <- c(possumNPH1$summaryRsq,
              possumNPH1_noage$summaryRsq,
              #possumNPH1_nobmi$summaryRsq,
              possumNPH1_noedu$summaryRsq,
              possumNPH1_noincome$summaryRsq
              #possumNPH1_nopovlev$summaryRsq
              #possumNPH1_nosex$summaryRsq,
              #possumNPH1_nosmoke$summaryRsq
)
legendVecNPH1 <- c("All", "Without age", #"Without bmi", 
                 "Without edu", "Without income" 
                 #,"Without povlev" #, "Without sex" #, "Without smoke"
)

pltNPH1 <- plt_rsq(rsq_df(RsqVecNPH1, legendVecNPH1))


# For NPH category 2
lambda_hat <- rowMeans(lambdaNPH_samples[, 2, ]) %>% as.numeric()

possumNPH2 <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula, trnX)
possumNPH2_noage <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_noage, trnX)  
possumNPH2_nosmoke <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_nosmoke, trnX)  
possumNPH2_noedu <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_noedu, trnX)
possumNPH2_noincome <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula, trnX)
possumNPH2_nopovlev <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_nopovlev, trnX)
possumNPH2_nosex <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_nosex, trnX)
possumNPH2_nobmi <- pos_summary(lambdaNPH_samples[, 2, ], lambda_hat, formula_nobmi, trnX)

add_sumNPH2 <- additive_summary_plot_2(possumNPH2) + xlab("") + theme_bw()

RsqVecNPH2 <- c(possumNPH2$summaryRsq,
                possumNPH2_noage$summaryRsq,
                #possumNPH2_nobmi$summaryRsq,
                possumNPH2_noedu$summaryRsq,
                possumNPH2_noincome$summaryRsq
                #possumNPH2_nopovlev$summaryRsq
                #possumNPH2_nosex$summaryRsq,
                #possumNPH2_nosmoke$summaryRsq
)
legendVecNPH2 <- c("All", "Without age", #"Without bmi", 
                   "Without edu", "Without income" 
                   #,"Without povlev" #, "Without sex" #, "Without smoke"
)

pltNPH2 <- plt_rsq(rsq_df(RsqVecNPH2, legendVecNPH2))

# For NPH category 3
lambda_hat <- rowMeans(lambdaNPH_samples[, 3, ]) %>% as.numeric()

possumNPH3 <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula, trnX)
possumNPH3_noage <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_noage, trnX)  
possumNPH3_nosmoke <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_nosmoke, trnX)  
possumNPH3_noedu <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_noedu, trnX)
possumNPH3_noincome <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula, trnX)
possumNPH3_nopovlev <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_nopovlev, trnX)
possumNPH3_nosex <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_nosex, trnX)
possumNPH3_nobmi <- pos_summary(lambdaNPH_samples[, 3, ], lambda_hat, formula_nobmi, trnX)

add_sumNPH3 <- additive_summary_plot_2(possumNPH3) + xlab("") + theme_bw()

RsqVecNPH3 <- c(possumNPH3$summaryRsq,
                possumNPH3_noage$summaryRsq,
                #possumNPH3_nobmi$summaryRsq,
                possumNPH3_noedu$summaryRsq,
                possumNPH3_noincome$summaryRsq
                #possumNPH3_nopovlev$summaryRsq
                #possumNPH3_nosex$summaryRsq,
                #possumNPH3_nosmoke$summaryRsq
)
legendVecNPH3 <- c("All", "Without age", #"Without bmi", 
                   "Without edu", "Without income" 
                   #,"Without povlev" #, "Without sex" #, "Without smoke"
)

pltNPH3 <- plt_rsq(rsq_df(RsqVecNPH3, legendVecNPH3))

# For NPH category 4
lambda_hat <- rowMeans(lambdaNPH_samples[, 4, ]) %>% as.numeric()

possumNPH4 <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula, trnX)
possumNPH4_noage <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_noage, trnX)  
possumNPH4_nosmoke <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_nosmoke, trnX)  
possumNPH4_noedu <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_noedu, trnX)
possumNPH4_noincome <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula, trnX)
possumNPH4_nopovlev <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_nopovlev, trnX)
possumNPH4_nosex <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_nosex, trnX)
possumNPH4_nobmi <- pos_summary(lambdaNPH_samples[, 4, ], lambda_hat, formula_nobmi, trnX)

add_sumNPH4 <- additive_summary_plot_2(possumNPH4) + xlab("") + theme_bw()

RsqVecNPH4 <- c(possumNPH4$summaryRsq,
                possumNPH4_noage$summaryRsq,
                #possumNPH4_nobmi$summaryRsq,
                possumNPH4_noedu$summaryRsq,
                possumNPH4_noincome$summaryRsq
                #possumNPH4_nopovlev$summaryRsq
                #possumNPH4_nosex$summaryRsq,
                #possumNPH4_nosmoke$summaryRsq
)
legendVecNPH4 <- c("All", "Without age", #"Without bmi", 
                   "Without edu", "Without income" 
                   #,"Without povlev" #, "Without sex" #, "Without smoke"
)

pltNPH4 <- plt_rsq(rsq_df(RsqVecNPH4, legendVecNPH4))

# Results -- Plots
add_sumPH + add_sumNPH1
add_sumPH + add_sumNPH2
add_sumPH + add_sumNPH3
add_sumPH + add_sumNPH4

(add_sumNPH1 + add_sumNPH2) / (add_sumNPH3 + add_sumNPH4)

pltPH / (pltNPH1 + pltNPH2) / (pltNPH3 + pltNPH4)

(pltNPH1 + pltNPH2) / (pltNPH3 + pltNPH4)

# CL modeling based on maximum likelihood

meps_lm <- meps %>% as.data.frame() %>% 
  mutate(sex = as.factor(sex),
         marital = as.factor(marital),
         race = as.factor(race),
         region = as.factor(region),
         seatbelt = as.factor(seatbelt),
         smoke = as.factor(smoke),
         phealth = ordered(phealth))

phealth_tst <- meps_lm[-trnIndx, ]$phealth

formula_lm <- phealth ~ age + bmi + edu + income + povlev + sex + marital + race + seatbelt + smoke

fit_cloglog <- vglm(formula = formula_lm, family = cumulative("clogloglink", parallel = TRUE), 
            data = meps_lm[trnIndx, ] %>% as.data.frame())

fit_probit <- vglm(formula = formula_lm, family = cumulative("probitlink", parallel = TRUE), 
            data = meps_lm[trnIndx, ] %>% as.data.frame())

phat_cloglog <- predict(fit_cloglog, meps_lm[-trnIndx, ], type = "response")

pred_probCloglog <- sapply(1:length(phealth_tst), \(i) phat_cloglog[i, phealth_tst[i]])

phat_probit <- predict(fit_probit, meps_lm[-trnIndx, ], type = "response")

pred_probProbit <- sapply(1:length(phealth_tst), \(i) phat_probit[i, phealth_tst[i]])

dat <- data.frame(x = rep(1:nrow(tstX), 4), 
                  y = c(exp(pred_probPH), exp(pred_probNPH), pred_probCloglog, pred_probProbit), 
                  grp = rep(c("BART-Cloglog-PH", "BART-Cloglog-NPH", "LM-Cloglog", "LM-Probit"), each = nrow(tstX)))
dat$grp <- factor(dat$grp, levels = c("BART-Cloglog-PH", "BART-Cloglog-NPH", "LM-Cloglog", "LM-Probit")) 

ggplot() +
  geom_density(data = dat, mapping = aes(x = y, fill = grp), alpha = 0.4) + 
  labs(x = "predictive prob") +
  theme_bw() +
  theme(legend.position = "bottom")

t.test(pred_probNPH, log(pred_probProbit), paired = TRUE)
