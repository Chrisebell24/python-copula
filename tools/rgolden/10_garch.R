#!/usr/bin/env Rscript
# Reference GARCH fits from rugarch, for the margins of rcopula.garch.
#
# Each case simulates a fixed series with ugarchpath and fits it with
# ugarchfit; the Python parity test refits the same series with fit_garch and
# compares parameters and log-likelihood. Requires rugarch:
#   install.packages("rugarch", repos = "https://cloud.r-project.org")
#
# Conventions to keep in mind when comparing:
# * rugarch writes the ARMA mean around its long-run level,
#   r_t - mu = ar1 (r_{t-1} - mu) + ma1 eps_{t-1} + eps_t, so its `mu` is the
#   unconditional mean; rcopula's intercept is mu * (1 - ar1).
# * rugarch starts the variance recursion at mean(eps^2) of the current
#   residuals; rcopula uses the sample variance of the data. The two differ
#   only through the first few terms, so parameters agree closely and the
#   log-likelihoods to a small fraction of a unit.
suppressPackageStartupMessages(library(rugarch))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

one_case <- function(model, arma, dist, truth, n, seed) {
  spec_true <- ugarchspec(
    variance.model = list(model = model, garchOrder = c(1, 1)),
    mean.model = list(armaOrder = arma, include.mean = TRUE),
    distribution.model = dist,
    fixed.pars = truth
  )
  path <- ugarchpath(spec_true, n.sim = n, n.start = 500, m.sim = 1, rseed = seed)
  x <- as.numeric(fitted(path))

  spec_fit <- ugarchspec(
    variance.model = list(model = model, garchOrder = c(1, 1)),
    mean.model = list(armaOrder = arma, include.mean = TRUE),
    distribution.model = dist
  )
  fit <- ugarchfit(spec_fit, x, solver = "hybrid")
  list(
    model = model, arma = arma, dist = dist, truth = as.list(truth),
    x = x,
    coef = as.list(coef(fit)),
    loglik = likelihood(fit),
    sigma_tail = tail(as.numeric(sigma(fit)), 100)
  )
}

res <- list(
  sgarch_norm = one_case(
    "sGARCH", c(0, 0), "norm",
    c(mu = 0.05, omega = 0.05, alpha1 = 0.10, beta1 = 0.85), 3000, 101
  ),
  gjr_norm = one_case(
    "gjrGARCH", c(0, 0), "norm",
    c(mu = 0.03, omega = 0.04, alpha1 = 0.03, beta1 = 0.87, gamma1 = 0.12), 3000, 202
  ),
  sgarch_arma11_norm = one_case(
    "sGARCH", c(1, 1), "norm",
    c(mu = 0.10, ar1 = 0.5, ma1 = -0.2, omega = 0.05, alpha1 = 0.08, beta1 = 0.88),
    3000, 303
  ),
  gjr_ar1_std = one_case(
    "gjrGARCH", c(1, 0), "std",
    c(mu = 0.05, ar1 = 0.2, omega = 0.04, alpha1 = 0.03, beta1 = 0.87, gamma1 = 0.12,
      shape = 6),
    3000, 404
  ),
  `_meta` = list(r_version = R.version.string,
                 rugarch_version = as.character(packageVersion("rugarch")))
)

write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "garch.json"))
cat("wrote", file.path(outdir, "garch.json"), "\n")
