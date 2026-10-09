#!/usr/bin/env Rscript
# Reference values in four and five dimensions.
#
# Every other fixture stops at d <= 3 for the multi-parameter families, and in
# three dimensions the row-major and column-major orders of the lower triangle
# coincide -- (2,1), (3,1), (3,2) either way. A correlation vector read in the
# wrong order therefore passes every d <= 3 check; that is how the p2P/P2p bug
# fixed in 0.2.0 survived. From d = 4 the orders differ, so this script uses
# correlation vectors whose entries are all distinct and records R's
# `getSigma()` next to each one.
#
# Original work for rcopula; it only *calls* the R copula package to record its
# outputs. See NOTICE for the clean-room policy.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(mvtnorm))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
maybe <- function(expr) tryCatch(expr, error = function(e) NA_real_)

set.seed(20261008)
grid_u <- function(n, d) matrix(runif(n * d), nrow = n, ncol = d)

# Do NOT use pCopula's default algorithm for the elliptical CDFs: for
# 3 <= d <= 5 it is Miwa with 128 steps (~1e-4 error). A tight Genz-Bretz keeps
# the fixture from being the less accurate side; see test_golden_elliptical.py.
ell_cdf <- function(u, cop) {
  pCopula(u, cop, algorithm = GenzBretz(maxpts = 250000, abseps = 1e-8))
}

# Correlation vectors in R's (column-major, P2p) order. All entries distinct, so
# reading them in any other order yields a different matrix.
un4 <- c(0.62, -0.18, 0.35, 0.47, 0.11, -0.29)
un5 <- c(0.55, 0.32, -0.21, 0.14, 0.41, 0.08, -0.12, 0.27, 0.36, 0.61)

res <- list()

# ---------------------------------------------------------------------------
# Elliptical: density, CDF, tau/rho, Sigma at fixed points
# ---------------------------------------------------------------------------
ell_specs <- list(
  list(dim = 4, dispstr = "un",   rho = un4),
  list(dim = 5, dispstr = "un",   rho = un5),
  list(dim = 4, dispstr = "toep", rho = c(0.5, 0.25, -0.1)),
  list(dim = 5, dispstr = "toep", rho = c(0.6, 0.35, 0.15, -0.05)),
  list(dim = 4, dispstr = "ar1",  rho = 0.55),
  list(dim = 5, dispstr = "ar1",  rho = -0.4)
)
for (spec in ell_specs) {
  u <- grid_u(20, spec$dim)
  for (fam in c("normal", "t")) {
    # pCopula for t needs an integer df.
    df <- if (fam == "t") (if (spec$dim == 4) 5 else 3) else NA_real_
    cop <- if (fam == "normal") normalCopula(spec$rho, dim = spec$dim, dispstr = spec$dispstr)
           else tCopula(spec$rho, dim = spec$dim, dispstr = spec$dispstr, df = df)
    key <- sprintf("ell_%s_d%d_%s", fam, spec$dim, spec$dispstr)
    res[[key]] <- list(
      kind = "elliptical", family = fam, dim = spec$dim, dispstr = spec$dispstr,
      rho = spec$rho, df = df, u = u,
      sigma = getSigma(cop),
      pdf = dCopula(u, cop), logpdf = dCopula(u, cop, log = TRUE),
      cdf = ell_cdf(u, cop),
      tau = maybe(tau(cop)),
      # R has no Spearman's rho for the t copula; NA there.
      rho_s = maybe(rho(cop))
    )
  }
}

# ---------------------------------------------------------------------------
# Fits on exported draws. The sample is written to the fixture, so both sides
# estimate from identical data and the estimates are deterministic.
# ---------------------------------------------------------------------------
#
# The draws are converted to pseudo-observations first. Both sides would take
# raw uniforms as pseudo-observations as they are, but the mpl variance is
# derived for ranks, and its empirical ingredients are computed differently by
# the two implementations when the input is not a set of ranks (R and rcopula
# then disagree by ~10% on a Clayton standard error while agreeing to 1e-3 on
# genuine pseudo-observations).
add_fit <- function(key, cop, u, methods, extra = list()) {
  u <- pobs(u)
  entry <- c(list(kind = "fit", u = u), extra)
  for (m in methods) {
    f <- maybe(fitCopula(cop, u, method = m))
    if (identical(f, NA_real_)) next
    s <- maybe(summary(f)$coefficients)
    entry[[paste0("est_", m)]] <- maybe(as.numeric(coef(f)))
    entry[[paste0("se_", m)]] <- if (identical(s, NA_real_)) NA_real_ else as.numeric(s[, 2])
    entry[[paste0("loglik_", m)]] <- maybe(as.numeric(f@loglik))
  }
  res[[key]] <<- entry
}

n_fit <- 1000
u <- rCopula(n_fit, normalCopula(un4, dim = 4, dispstr = "un"))
add_fit("fit_normal_d4_un", normalCopula(dim = 4, dispstr = "un"), u,
        c("itau", "irho", "mpl"),
        list(family = "normal", dim = 4, dispstr = "un", truth = un4))

u <- rCopula(n_fit, normalCopula(un5, dim = 5, dispstr = "un"))
add_fit("fit_normal_d5_un", normalCopula(dim = 5, dispstr = "un"), u,
        c("itau", "irho", "mpl"),
        list(family = "normal", dim = 5, dispstr = "un", truth = un5))

u <- rCopula(n_fit, tCopula(un4, dim = 4, dispstr = "un", df = 5))
add_fit("fit_t_d4_un", tCopula(dim = 4, dispstr = "un"), u, c("itau.mpl", "mpl"),
        list(family = "t", dim = 4, dispstr = "un", truth = c(un4, 5)))

u <- rCopula(n_fit, tCopula(un5, dim = 5, dispstr = "un", df = 4))
add_fit("fit_t_d5_un", tCopula(dim = 5, dispstr = "un"), u, c("itau.mpl", "mpl"),
        list(family = "t", dim = 5, dispstr = "un", truth = c(un5, 4)))
# Same sample, df held at its true value: plain inversion of tau.
add_fit("fit_t_d5_un_dffixed", tCopula(dim = 5, dispstr = "un", df = 4, df.fixed = TRUE), u,
        c("itau", "mpl"),
        list(family = "t_dffixed", dim = 5, dispstr = "un", df = 4, truth = un5))

u <- rCopula(n_fit, normalCopula(c(0.6, 0.35, 0.15, -0.05), dim = 5, dispstr = "toep"))
add_fit("fit_normal_d5_toep", normalCopula(dim = 5, dispstr = "toep"), u,
        c("itau", "irho", "mpl"),
        list(family = "normal", dim = 5, dispstr = "toep", truth = c(0.6, 0.35, 0.15, -0.05)))

u <- rCopula(n_fit, normalCopula(0.55, dim = 4, dispstr = "ar1"))
add_fit("fit_normal_d4_ar1", normalCopula(dim = 4, dispstr = "ar1"), u,
        c("itau", "irho", "mpl"),
        list(family = "normal", dim = 4, dispstr = "ar1", truth = 0.55))

u <- rCopula(n_fit, normalCopula(0.3, dim = 5, dispstr = "ex"))
add_fit("fit_normal_d5_ex", normalCopula(dim = 5, dispstr = "ex"), u,
        c("itau", "irho", "mpl"),
        list(family = "normal", dim = 5, dispstr = "ex", truth = 0.3))

u <- rCopula(n_fit, claytonCopula(2, dim = 4))
add_fit("fit_clayton_d4", claytonCopula(dim = 4), u, c("itau", "mpl"),
        list(family = "clayton", dim = 4, truth = 2))

u <- rCopula(n_fit, gumbelCopula(1.8, dim = 5))
add_fit("fit_gumbel_d5", gumbelCopula(dim = 5), u, c("itau", "mpl"),
        list(family = "gumbel", dim = 5, truth = 1.8))

# ---------------------------------------------------------------------------
# Archimedean density and CDF in d = 4, 5 (AMH via onacopula: R's amhCopula is
# restricted to d = 2, the nacopula implementation is not)
# ---------------------------------------------------------------------------
arch <- list(
  clayton = list(ctor = function(th, d) claytonCopula(th, dim = d), thetas = c(0.05, 1.3, 6)),
  gumbel  = list(ctor = function(th, d) gumbelCopula(th, dim = d),  thetas = c(1.02, 1.7, 4.5)),
  frank   = list(ctor = function(th, d) frankCopula(th, dim = d),   thetas = c(0.2, 3.5, 12)),
  joe     = list(ctor = function(th, d) joeCopula(th, dim = d),     thetas = c(1.03, 1.9, 5)),
  amh     = list(ctor = function(th, d) onacopulaL("AMH", list(th, seq_len(d))), thetas = c(0.05, 0.4, 0.85))
)
for (fam in names(arch)) {
  for (d in c(4, 5)) {
    u <- grid_u(25, d)
    for (th in arch[[fam]]$thetas) {
      cop <- arch[[fam]]$ctor(th, d)
      res[[sprintf("arch_%s_d%d_theta%s", fam, d, format(th))]] <- list(
        kind = "archimedean", family = fam, dim = d, theta = th, u = u,
        pdf = dCopula(u, cop), logpdf = dCopula(u, cop, log = TRUE),
        cdf = pCopula(u, cop)
      )
    }
  }
}

# ---------------------------------------------------------------------------
# Nested Archimedean. R's dCopula supports only non-nested Archimedean copulas,
# so only the CDF has a reference. Components are 1-based here.
# ---------------------------------------------------------------------------
nested <- list(
  clayton_d5 = list(family = "Clayton", root = 0.8, root_comp = integer(0),
                    children = list(list(theta = 2.5, comp = c(1, 2)),
                                    list(theta = 4, comp = c(3, 4, 5)))),
  gumbel_d4  = list(family = "Gumbel", root = 1.4, root_comp = c(1),
                    children = list(list(theta = 2.6, comp = c(2, 3, 4)))),
  frank_d5   = list(family = "Frank", root = 1.5, root_comp = c(5),
                    children = list(list(theta = 4, comp = c(1, 2)),
                                    list(theta = 6, comp = c(3, 4)))),
  joe_d4     = list(family = "Joe", root = 1.3, root_comp = integer(0),
                    children = list(list(theta = 2, comp = c(1, 3)),
                                    list(theta = 3, comp = c(2, 4))))
)
for (nm in names(nested)) {
  spec <- nested[[nm]]
  # onacopulaL takes the tree as a plain list; C() would resolve to stats::C here.
  kids <- lapply(spec$children, function(ch) list(ch$theta, ch$comp))
  cop <- onacopulaL(spec$family, list(spec$root, spec$root_comp, kids))
  d <- dim(cop)
  u <- grid_u(25, d)
  res[[paste0("nested_", nm)]] <- list(
    kind = "nested", family = tolower(spec$family), dim = d, root = spec$root,
    root_comp = spec$root_comp - 1L,
    children = lapply(spec$children, function(ch) list(theta = ch$theta, comp = ch$comp - 1L)),
    u = u, cdf = pCopula(u, cop), pdf = maybe(dCopula(u, cop))
  )
}

# ---------------------------------------------------------------------------
# Marginal copulas of a 5-d unstructured elliptical copula. margCopula keeps
# coordinates in increasing order (a logical mask).
# ---------------------------------------------------------------------------
for (fam in c("normal", "t")) {
  full <- if (fam == "normal") normalCopula(un5, dim = 5, dispstr = "un")
          else tCopula(un5, dim = 5, dispstr = "un", df = 3.5)
  for (keep in list(c(1, 3, 4), c(2, 5), c(1, 2, 4, 5))) {
    mask <- seq_len(5) %in% keep
    mc <- margCopula(full, mask)
    u <- grid_u(15, length(keep))
    res[[sprintf("marg_%s_%s", fam, paste(keep, collapse = ""))]] <- list(
      kind = "marginal", family = fam, dim = 5, rho = un5,
      df = if (fam == "t") 3.5 else NA_real_,
      keep = keep - 1L, params = as.numeric(mc@parameters), sigma = getSigma(mc),
      u = u, pdf = dCopula(u, mc), logpdf = dCopula(u, mc, log = TRUE)
    )
  }
}

# ---------------------------------------------------------------------------
# Rosenblatt transform (cCopula) and its inverse, d = 4
# ---------------------------------------------------------------------------
ros <- list(
  normal_un  = normalCopula(un4, dim = 4, dispstr = "un"),
  t_un       = tCopula(un4, dim = 4, dispstr = "un", df = 5),
  normal_toep = normalCopula(c(0.5, 0.25, -0.1), dim = 4, dispstr = "toep"),
  clayton    = claytonCopula(1.3, dim = 4),
  gumbel     = gumbelCopula(1.7, dim = 4),
  frank      = frankCopula(3.5, dim = 4),
  joe        = joeCopula(1.9, dim = 4)
)
u <- grid_u(20, 4)
for (nm in names(ros)) {
  cop <- ros[[nm]]
  isEll <- is(cop, "ellipCopula")
  res[[paste0("ros_", nm)]] <- list(
    kind = "rosenblatt", name = nm, dim = 4, u = u,
    family = if (isEll) (if (is(cop, "tCopula")) "t" else "normal") else sub("_.*", "", nm),
    dispstr = if (isEll) cop@dispstr else NA_character_,
    params = as.numeric(cop@parameters),
    df = if (is(cop, "tCopula")) 5 else NA_real_,
    forward = cCopula(u, cop),
    inverse = maybe(cCopula(u, cop, inverse = TRUE))
  )
}

res[["_meta"]] <- list(
  r_version      = R.version.string,
  copula_version = as.character(packageVersion("copula"))
)
write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "highdim.json"))
cat("wrote", file.path(outdir, "highdim.json"), "\n")
