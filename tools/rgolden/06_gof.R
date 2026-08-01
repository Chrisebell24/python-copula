#!/usr/bin/env Rscript
# Reference goodness-of-fit values. The data are committed with the fixtures so
# both sides test the same sample.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(20260801)
x <- rCopula(300, claytonCopula(4))
fams <- list(clayton = claytonCopula(), gumbel = gumbelCopula(),
             frank = frankCopula(), normal = normalCopula())

sn <- list(); p_mult <- list(); p_pb <- list()
for (nm in names(fams)) {
  set.seed(1); g <- gofCopula(fams[[nm]], x, N = 500, simulation = "pb", verbose = FALSE)
  set.seed(1); m <- gofCopula(fams[[nm]], x, N = 500, simulation = "mult", verbose = FALSE)
  sn[[nm]] <- as.numeric(g$statistic)
  p_pb[[nm]] <- as.numeric(g$p.value)
  p_mult[[nm]] <- as.numeric(m$p.value)
}

write(toJSON(list(data = x, Sn = sn, p_pb = p_pb, p_mult = p_mult,
                  `_meta` = list(r_version = R.version.string,
                                 copula_version = as.character(packageVersion("copula")))),
             digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "gof.json"))
cat("wrote", file.path(outdir, "gof.json"), "\n")
