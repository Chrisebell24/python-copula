#!/usr/bin/env Rscript
# Reference values for the non-Archimedean, non-elliptical families.
# Original work for rcopula; only *calls* the R copula package. See NOTICE.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Several of these families lack some methods in R (fgmCopula has no `lambda`,
# for instance). Record NA rather than aborting.
maybe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

set.seed(20260731)
u <- matrix(runif(60 * 2), ncol = 2)

res <- list()

add <- function(key, cop, extra = list()) {
  res[[key]] <<- c(list(
    u = u,
    pdf = maybe(dCopula(u, cop)),
    logpdf = maybe(dCopula(u, cop, log = TRUE)),
    cdf = maybe(pCopula(u, cop)),
    tau = maybe(tau(cop)),
    rho = maybe(rho(cop)),
    lambdaL = maybe(lambda(cop)[["lower"]]),
    lambdaU = maybe(lambda(cop)[["upper"]])
  ), extra)
}

for (th in c(0.1, 0.5, 1, 2, 3, 10, 50)) {
  add(sprintf("plackett_%s", format(th)), plackettCopula(th),
      list(family = "plackett", theta = th))
}
for (th in c(-1, -0.5, 0, 0.3, 0.7, 1)) {
  add(sprintf("fgm_%s", format(th)), fgmCopula(th), list(family = "fgm", theta = th))
}
for (a in list(c(0.2, 0.8), c(0.5, 0.5), c(0.9, 0.1), c(0.3, 0.3))) {
  add(sprintf("mo_%s_%s", format(a[1]), format(a[2])), moCopula(a),
      list(family = "mo", alpha = a))
}
add("indep", indepCopula(2), list(family = "indep"))
add("fh_upper", upfhCopula(2), list(family = "fh_upper"))
add("fh_lower", lowfhCopula(2), list(family = "fh_lower"))

res[["_meta"]] <- list(r_version = R.version.string,
                       copula_version = as.character(packageVersion("copula")))
write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "other.json"))
cat("wrote", file.path(outdir, "other.json"), "\n")
