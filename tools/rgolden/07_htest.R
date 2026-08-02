#!/usr/bin/env Rscript
# Reference statistics for the specification tests. Only the *statistics* are
# recorded: rcopula generates null distributions by randomisation rather than by
# R's multiplier bootstrap, so the p-values are not comparable.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(20260801)
fams <- list(clayton = claytonCopula(3), frank = frankCopula(5),
             gumbel = gumbelCopula(3), normal = normalCopula(0.6))

cases <- list()
for (nm in names(fams)) {
  x <- rCopula(400, fams[[nm]])
  set.seed(1); e <- exchTest(x, N = 100)
  set.seed(1); r <- radSymTest(x, N = 100)
  cases[[nm]] <- list(data = x,
                      exch_stat = as.numeric(e$statistic),
                      radsym_stat = as.numeric(r$statistic))
}

write(toJSON(list(cases = cases,
                  `_meta` = list(r_version = R.version.string,
                                 copula_version = as.character(packageVersion("copula")))),
             digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "htest.json"))
cat("wrote", file.path(outdir, "htest.json"), "\n")
