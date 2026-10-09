#!/usr/bin/env Rscript
# Reference values for regular vines, the BB1/BB7 families and rotated
# pair-copulas, from R's VineCopula package.
# Original work for rcopula; only *calls* the R VineCopula package. See NOTICE.
#
# Needs VineCopula (install.packages("VineCopula")) besides jsonlite.
suppressPackageStartupMessages(library(VineCopula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
res <- list()

# ---------------------------------------------------------------------------
# 1. Bivariate families: BB1 (7), BB7 (9) and rotations of Clayton (3),
#    Gumbel (4), Joe (6), BB1 and BB7. VineCopula's codes: 1x = 180 degrees,
#    2x = 90 degrees (reflects the FIRST argument), 3x = 270 degrees (reflects
#    the SECOND argument); 90/270 take negated parameters.
# ---------------------------------------------------------------------------
set.seed(20261008)
u <- matrix(runif(40 * 2, 0.01, 0.99), ncol = 2)
pairs <- list()
add_pair <- function(family, par, par2 = 0) {
  key <- sprintf("f%d_%s_%s", family, format(par), format(par2))
  pairs[[key]] <<- list(
    family = family, par = par, par2 = par2, u = u,
    pdf = BiCopPDF(u[, 1], u[, 2], family, par, par2),
    cdf = BiCopCDF(u[, 1], u[, 2], family, par, par2),
    # hfunc1 = P(U2 <= u2 | U1 = u1); hfunc2 = P(U1 <= u1 | U2 = u2)
    hfunc1 = BiCopHfunc1(u[, 1], u[, 2], family, par, par2),
    hfunc2 = BiCopHfunc2(u[, 1], u[, 2], family, par, par2),
    tau = BiCopPar2Tau(family, par, par2),
    lambda = unname(unlist(BiCopPar2TailDep(family, par, par2)))
  )
}
for (p in list(c(0.5, 1.5), c(2.0, 1.0), c(0.2, 3.5), c(3.0, 1.2))) add_pair(7, p[1], p[2])
for (p in list(c(1.5, 2.0), c(1.0, 2.0), c(3.0, 0.4), c(1.2, 5.0))) add_pair(9, p[1], p[2])
for (f in c(13, 14, 16, 17, 19)) {
  add_pair(f, switch(as.character(f %% 10), "3" = 2.0, "4" = 1.8, "6" = 1.6, "7" = 0.6, "9" = 1.4),
           switch(as.character(f %% 10), "7" = 1.5, "9" = 1.2, 0))
}
for (f in c(23, 24, 26, 27, 29, 33, 34, 36, 37, 39)) {
  add_pair(f, -switch(as.character(f %% 10), "3" = 2.0, "4" = 1.8, "6" = 1.6, "7" = 0.6, "9" = 1.4),
           -switch(as.character(f %% 10), "7" = 1.5, "9" = 1.2, 0))
}
res[["pairs"]] <- pairs

# ---------------------------------------------------------------------------
# 2. A five-dimensional R-vine that is neither a C- nor a D-vine (the example
#    matrix of VineCopula's RVineMatrix help page), with mixed families.
# ---------------------------------------------------------------------------
M <- matrix(c(5, 2, 3, 1, 4,
              0, 2, 3, 4, 1,
              0, 0, 3, 4, 1,
              0, 0, 0, 4, 1,
              0, 0, 0, 0, 1), 5, 5)
fam <- matrix(0, 5, 5); par <- matrix(0, 5, 5); par2 <- matrix(0, 5, 5)
# tree 1 (row 5)
fam[5, 1] <- 3;  par[5, 1] <- 2.0
fam[5, 2] <- 24; par[5, 2] <- -1.8
fam[5, 3] <- 7;  par[5, 3] <- 0.6; par2[5, 3] <- 1.5
fam[5, 4] <- 2;  par[5, 4] <- 0.5; par2[5, 4] <- 5
# tree 2 (row 4)
fam[4, 1] <- 9;  par[4, 1] <- 1.4; par2[4, 1] <- 1.2
fam[4, 2] <- 5;  par[4, 2] <- -3
fam[4, 3] <- 33; par[4, 3] <- -1.5
# tree 3 (row 3)
fam[3, 1] <- 1;  par[3, 1] <- 0.3
fam[3, 2] <- 16; par[3, 2] <- 1.6
# tree 4 (row 2)
fam[2, 1] <- 6;  par[2, 1] <- 1.3
RVM <- RVineMatrix(M, fam, par, par2)
set.seed(20261009)
uv <- matrix(runif(50 * 5, 0.02, 0.98), ncol = 5)
res[["rvine_density"]] <- list(
  matrix = M, family = fam, par = par, par2 = par2, u = uv,
  pdf = RVinePDF(uv, RVM),
  loglik = RVineLogLik(uv, RVM, separate = TRUE)$loglik,
  pit = RVinePIT(uv, RVM)
)

# ---------------------------------------------------------------------------
# 3. Dissmann structure selection on data simulated from that vine.
# ---------------------------------------------------------------------------
set.seed(20261010)
sim <- RVineSim(600, RVM)
pu <- pobs(sim)
familyset <- c(1, 2, 3, 4, 5, 6, 13, 14, 16, 23, 24, 26, 33, 34, 36)
sel <- RVineStructureSelect(pu, familyset = familyset, type = 0,
                            selectioncrit = "AIC", indeptest = FALSE)
# Tree-1 edges: column i, row d of the matrix joins M[d, i] and M[i, i].
d <- ncol(pu)
tree1 <- t(sapply(1:(d - 1), function(i) sort(c(sel$Matrix[d, i], sel$Matrix[i, i]))))
res[["selection"]] <- list(
  u = pu, familyset = familyset,
  matrix = sel$Matrix, family = sel$family, par = sel$par, par2 = sel$par2,
  tree1 = tree1,
  loglik = RVineLogLik(pu, sel)$loglik
)

res[["_meta"]] <- list(r_version = R.version.string,
                       VineCopula_version = as.character(packageVersion("VineCopula")))
write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null", matrix = "rowmajor"),
      file.path(outdir, "vine.json"))
cat("wrote", file.path(outdir, "vine.json"), "\n")
