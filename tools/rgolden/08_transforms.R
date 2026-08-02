#!/usr/bin/env Rscript
# Reference Rosenblatt transforms. R's cCopula covers Archimedean and elliptical
# families only, which is exactly the set rcopula implements analytically.
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

outdir <- file.path("tests", "golden")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(20260801)
u <- rCopula(200, claytonCopula(3, dim = 3))

res <- list(u = u,
            clayton = cCopula(u, claytonCopula(3, dim = 3)),
            gumbel  = cCopula(u, gumbelCopula(2.5, dim = 3)),
            frank   = cCopula(u, frankCopula(5, dim = 3)),
            normal  = cCopula(u, normalCopula(0.5, dim = 3)),
            t       = cCopula(u, tCopula(0.5, dim = 3, df = 5)),
            `_meta` = list(r_version = R.version.string,
                           copula_version = as.character(packageVersion("copula"))))

write(toJSON(res, digits = I(17), auto_unbox = TRUE, null = "null"),
      file.path(outdir, "transforms.json"))
cat("wrote", file.path(outdir, "transforms.json"), "\n")
