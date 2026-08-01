#!/usr/bin/env Rscript
# Emit reference values for the Archimedean families.
#
# This script is original work for rcopula; it only *calls* the R copula
# package to record its outputs. See NOTICE for the clean-room policy.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(mvtnorm))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# A fixed, reproducible evaluation grid in the open unit cube.
set.seed(20260731)
grid_u <- function(n, d) matrix(runif(n * d), nrow = n, ncol = d)

families <- list(
  clayton = list(ctor = claytonCopula, thetas = c(0.5, 1, 2, 5, 10), dims = c(2, 3, 5)),
  gumbel  = list(ctor = gumbelCopula,  thetas = c(1.2, 1.5, 2, 4, 8), dims = c(2, 3, 5)),
  frank   = list(ctor = frankCopula,   thetas = c(0.5, 2, 5, 10, 20), dims = c(2, 3, 5)),
  joe     = list(ctor = joeCopula,     thetas = c(1.1, 1.5, 2, 3, 6),   dims = c(2, 3, 5)),
  # R's amhCopula is restricted to d = 2; rcopula supports d > 2, so only the
  # bivariate case can be cross-checked here.
  amh     = list(ctor = amhCopula,     thetas = c(-0.9, -0.5, 0.3, 0.7, 0.95), dims = c(2))
)

# Not every family implements every method in R -- `rho()` has no joeCopula
# method at all, for instance. Record NA rather than aborting the whole run;
# the Python side then skips that comparison.
maybe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

res <- list()
for (fam in names(families)) {
  spec <- families[[fam]]
  for (d in spec$dims) {
    u <- grid_u(50, d)
    for (th in spec$thetas) {
      cop <- spec$ctor(th, dim = d)
      key <- sprintf("%s_d%d_theta%s", fam, d, format(th))
      res[[key]] <- list(
        family = fam, dim = d, theta = th,
        u      = u,
        pdf    = dCopula(u, cop),
        logpdf = dCopula(u, cop, log = TRUE),
        cdf    = pCopula(u, cop),
        tau    = maybe(tau(cop)),
        rho    = maybe(rho(cop)),
        lambdaL = maybe(lambda(cop)[["lower"]]),
        lambdaU = maybe(lambda(cop)[["upper"]]),
        # generator and its inverse, on a separate 1-d grid
        psi_t   = seq(0.01, 5, length.out = 25),
        psi     = maybe(psi(cop, seq(0.01, 5, length.out = 25))),
        ipsi_u  = seq(0.01, 0.99, length.out = 25),
        ipsi    = maybe(iPsi(cop, seq(0.01, 0.99, length.out = 25)))
      )
    }
  }
}

# iTau / iRho inversion targets
inv <- list()
for (t in c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9)) {
  inv[[sprintf("clayton_itau_%s", t)]] <- iTau(claytonCopula(), t)
  inv[[sprintf("gumbel_itau_%s",  t)]] <- iTau(gumbelCopula(),  t)
  inv[[sprintf("frank_itau_%s",   t)]] <- iTau(frankCopula(),   t)
  inv[[sprintf("frank_itau_neg_%s", t)]] <- iTau(frankCopula(), -t)
  inv[[sprintf("frank_irho_%s",   t)]] <- iRho(frankCopula(),   t)
  inv[[sprintf("joe_itau_%s",     t)]] <- maybe(iTau(joeCopula(),     t))
}
res[["_inversions"]] <- inv
res[["_meta"]] <- list(
  r_version      = R.version.string,
  copula_version = as.character(packageVersion("copula"))
)

write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "archimedean.json"))
cat("wrote", file.path(outdir, "archimedean.json"), "\n")

# ---------------------------------------------------------------------------
# Elliptical families. Written to a separate file so the Archimedean fixtures
# stay stable when only this section changes.
# ---------------------------------------------------------------------------
set.seed(20260731)
ell <- list()
for (d in c(2, 3, 5)) {
  u <- grid_u(20, d)
  for (rho in c(-0.3, 0.2, 0.5, 0.8)) {
    # Exchangeable correlation must satisfy rho >= -1/(d-1) to stay positive
    # definite, so negative rho is only admissible in low dimensions.
    if (rho < -1 / (d - 1)) next
    for (fam in c("normal", "t")) {
      for (dsp in c("ex", "ar1")) {
        cop <- if (fam == "normal") ellipCopula("normal", param = rho, dim = d, dispstr = dsp)
               else ellipCopula("t", param = rho, dim = d, dispstr = dsp, df = 4)
        key <- sprintf("%s_d%d_%s_rho%s", fam, d, dsp, format(rho))
        ell[[key]] <- list(
          family = fam, dim = d, dispstr = dsp, rho = rho,
          df = if (fam == "t") 4 else NA_real_,
          u = u,
          pdf = dCopula(u, cop), logpdf = dCopula(u, cop, log = TRUE),
          # Do NOT use pCopula's default algorithm. For 3 <= d <= 5 it picks
          # Miwa() with 128 steps, which carries ~1e-4 error. Use TVPACK where
          # it applies (d <= 3, essentially exact) and a tight GenzBretz
          # otherwise, so the fixture is never the less accurate side.
          cdf = if (d <= 3) pCopula(u, cop, algorithm = TVPACK(abseps = 1e-14))
                else pCopula(u, cop, algorithm = GenzBretz(maxpts = 250000, abseps = 1e-8)),
          tau = maybe(tau(cop)), rho_s = maybe(rho(cop)),
          lambdaL = maybe(lambda(cop)[["lower"]]),
          lambdaU = maybe(lambda(cop)[["upper"]])
        )
      }
    }
  }
}
# Unstructured, d = 3
u3 <- grid_u(20, 3)
for (fam in c("normal", "t")) {
  cop <- if (fam == "normal") normalCopula(c(0.6, 0.3, 0.2), dim = 3, dispstr = "un")
         else tCopula(c(0.6, 0.3, 0.2), dim = 3, dispstr = "un", df = 5)
  ell[[sprintf("%s_d3_un", fam)]] <- list(
    family = fam, dim = 3, dispstr = "un", rho = c(0.6, 0.3, 0.2),
    df = if (fam == "t") 5 else NA_real_,
    u = u3, pdf = dCopula(u3, cop), logpdf = dCopula(u3, cop, log = TRUE),
    cdf = pCopula(u3, cop, algorithm = TVPACK(abseps = 1e-14)), tau = maybe(tau(cop)), rho_s = maybe(rho(cop)),
    lambdaL = maybe(lambda(cop)[["lower"]]), lambdaU = maybe(lambda(cop)[["upper"]])
  )
}
ell[["_meta"]] <- list(r_version = R.version.string,
                       copula_version = as.character(packageVersion("copula")))
write(toJSON(ell, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "elliptical.json"))
cat("wrote", file.path(outdir, "elliptical.json"), "\n")
