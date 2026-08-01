#!/usr/bin/env Rscript
# Reference values for the empirical copula and the copula+margins distribution.
# Original work for rcopula; only *calls* the R copula package. See NOTICE.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(20260801)
# A fixed sample, committed alongside the fixtures so both sides see exactly the
# same data -- the empirical copula is a function of the data, not of a family.
X <- rCopula(200, claytonCopula(2))
grid <- as.matrix(expand.grid(seq(0.05, 0.95, by = 0.1), seq(0.1, 0.9, by = 0.2)))
dimnames(grid) <- NULL

mvdc_x <- rbind(c(1.0, 0.5), c(-0.5, 0.2), c(3.0, 1.5), c(0.0, 0.05))
mc <- mvdc(claytonCopula(2), c("norm", "exp"),
           list(list(mean = 1, sd = 2), list(rate = 3)))

res <- list(
  data = X,
  grid = grid,
  Cn_none         = C.n(grid, X, smoothing = "none"),
  Cn_beta         = C.n(grid, X, smoothing = "beta"),
  Cn_checkerboard = C.n(grid, X, smoothing = "checkerboard"),
  mvdc_x = mvdc_x,
  pMvdc  = pMvdc(mvdc_x, mc),
  dMvdc  = dMvdc(mvdc_x, mc),
  `_meta` = list(r_version = R.version.string,
                 copula_version = as.character(packageVersion("copula")))
)

write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "empirical.json"))
cat("wrote", file.path(outdir, "empirical.json"), "\n")
