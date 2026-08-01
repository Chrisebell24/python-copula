#!/usr/bin/env bash
# Regenerate every golden fixture from R. Requires R plus the copula, lcopula,
# qrmtools, mvtnorm and jsonlite packages -- see tools/rgolden/00_setup.R.
#
# Fixtures are committed, so CI never runs this. Re-run it only when the R
# copula version changes or when new parity coverage is added, and commit the
# regenerated files together with the change that motivated them.
set -euo pipefail
cd "$(dirname "$0")/../.."
for script in tools/rgolden/0[1-9]_*.R; do
    echo "=== $script ==="
    Rscript "$script"
done
echo
echo "Golden fixtures regenerated. R and copula versions:"
Rscript -e 'cat(R.version.string, "| copula", as.character(packageVersion("copula")), "\n")'
