#!/usr/bin/env Rscript
# One-off installation of the R packages the fixture scripts need.
#
# `copula` requires the GSL C library and `nloptr` requires NLopt, neither of
# which R installs for you. On macOS: brew install gsl nlopt cmake
install.packages(
  c("copula", "mvtnorm", "jsonlite", "lcopula", "qrmtools"),
  repos = "https://cloud.r-project.org"
)
cat("copula:", as.character(packageVersion("copula")), "\n")
