#!/usr/bin/env Rscript
# Reference fits. The data are generated here and committed with the fixtures,
# so both sides estimate from exactly the same sample -- R's RNG stream cannot
# be reproduced in NumPy, but the fitted values are then fully deterministic.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
maybe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

set.seed(20260801)
res <- list()

add <- function(key, cop, u, methods, extra = list()) {
  entry <- c(list(u = u), extra)
  for (m in methods) {
    f <- maybe(fitCopula(cop, u, method = m))
    if (identical(f, NA_real_)) next
    s <- maybe(summary(f)$coefficients)
    entry[[paste0("est_", m)]] <- maybe(as.numeric(coef(f)))
    entry[[paste0("se_", m)]]  <- if (identical(s, NA_real_)) NA_real_ else as.numeric(s[, 2])
    entry[[paste0("loglik_", m)]] <- maybe(as.numeric(f@loglik))
  }
  res[[key]] <<- entry
}

one_par <- c("mpl", "ml", "itau", "irho")

u <- rCopula(800, claytonCopula(2));  add("clayton_2", claytonCopula(), u, one_par, list(family="clayton", truth=2))
u <- rCopula(800, gumbelCopula(2.5)); add("gumbel_2.5", gumbelCopula(), u, one_par, list(family="gumbel", truth=2.5))
u <- rCopula(800, frankCopula(5));    add("frank_5", frankCopula(), u, one_par, list(family="frank", truth=5))
u <- rCopula(800, joeCopula(3));      add("joe_3", joeCopula(), u, one_par, list(family="joe", truth=3))
u <- rCopula(800, normalCopula(0.6)); add("normal_0.6", normalCopula(), u, one_par, list(family="normal", truth=0.6))

# Multi-parameter, unstructured
u <- rCopula(1200, normalCopula(c(0.6, 0.3, 0.2), dim = 3, dispstr = "un"))
add("normal_un_d3", normalCopula(dim = 3, dispstr = "un"), u, c("mpl", "itau", "irho"),
    list(family = "normal_un", truth = c(0.6, 0.3, 0.2)))

u <- rCopula(1200, tCopula(c(0.5, 0.3, 0.4), dim = 3, dispstr = "un", df = 5))
add("t_un_d3", tCopula(dim = 3, dispstr = "un"), u, c("mpl", "itau.mpl"),
    list(family = "t_un", truth = c(0.5, 0.3, 0.4, 5)))

res[["_meta"]] <- list(r_version = R.version.string,
                       copula_version = as.character(packageVersion("copula")))
write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "fitting.json"))
cat("wrote", file.path(outdir, "fitting.json"), "\n")
