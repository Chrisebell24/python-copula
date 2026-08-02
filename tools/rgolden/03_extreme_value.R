#!/usr/bin/env Rscript
# Reference values for the extreme-value families.
# Original work for rcopula; only *calls* the R copula package. See NOTICE.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
maybe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

set.seed(20260731)
u <- matrix(runif(60 * 2), ncol = 2)
w <- seq(0.02, 0.98, length.out = 33)   # Pickands grid

res <- list()
add <- function(key, cop, extra) {
  res[[key]] <<- c(list(
    u = u, w = w,
    A = maybe(A(cop, w)),
    dAdu1 = maybe(dAdu(cop, w)[, 1]),
    dAdu2 = maybe(dAdu(cop, w)[, 2]),
    pdf = maybe(dCopula(u, cop)),
    logpdf = maybe(dCopula(u, cop, log = TRUE)),
    cdf = maybe(pCopula(u, cop)),
    tau = maybe(tau(cop)), rho = maybe(rho(cop)),
    lambdaL = maybe(lambda(cop)[["lower"]]),
    lambdaU = maybe(lambda(cop)[["upper"]])
  ), extra)
}

for (th in c(0.3, 1, 2, 5)) {
  add(sprintf("galambos_%s", format(th)), galambosCopula(th),
      list(family = "galambos", theta = th))
}
for (th in c(0.5, 1.5, 3)) {
  add(sprintf("huslerreiss_%s", format(th)), huslerReissCopula(th),
      list(family = "huslerreiss", theta = th))
}
for (th in c(0.2, 0.6, 1)) {
  add(sprintf("tawn_%s", format(th)), tawnCopula(th), list(family = "tawn", theta = th))
}
for (r in c(-0.3, 0.5, 0.8)) {
  add(sprintf("tev_%s", format(r)), tevCopula(r, df = 4),
      list(family = "tev", rho_par = r, df = 4))
}
for (th in c(1.5, 3)) {
  add(sprintf("gumbel_ev_%s", format(th)), gumbelCopula(th),
      list(family = "gumbel_ev", theta = th))
}

res[["_meta"]] <- list(r_version = R.version.string,
                       copula_version = as.character(packageVersion("copula")))
write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "extreme_value.json"))
cat("wrote", file.path(outdir, "extreme_value.json"), "\n")
