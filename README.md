[![Documentation Status](https://readthedocs.org/projects/veros-bgc/badge/?version=latest)](https://veros-bgc.readthedocs.io/en/latest/?badge=latest)

# Veros-BGC
A biogeochemistry plugin for Veros implementing the MOBI ecosystem model.

This version targets the current Veros state/plugin API and follows the
`O_npzd` equations in the bundled MOBI Fortran reference. The optional nitrogen
cycle ports the coupled `O_npzd_nitrogen` and required `O_npzd_o2` branches,
including nitrate, dissolved organic N/P, diazotrophs, nitrogen fixation, and
water-column and benthic denitrification. Optional DIC, alkalinity, air-sea CO2
exchange, and implicit calcite redistribution correspond to the `O_carbon` /
`O_npzd_alk` branches. Iron, prognostic CaCO3/coccolithophores, and isotope
options are not yet implemented.

The bundled `bgc_global_4deg` setup enables nitrogen and uses the parameter
defaults defined by MOBI's Fortran `mobi_init` routine, including a maximum
phytoplankton growth rate of 0.6 d⁻¹ and `bbio = 1.066`.
Restart files from the earlier phosphate-only setup are not compatible because
the nitrogen configuration adds five prognostic tracers.

## Development with uv

The repository targets Python 3.14 and pins the current stable release,
Python 3.14.3, in `.python-version`. uv installs the interpreter when needed and
uses `uv.lock` for reproducible development, testing, model runs, and plotting:

```bash
$ uv sync --locked
$ uv run --locked python -m unittest discover -s test -v
$ uv run --locked python -m scripts.run_bgc_1yr --days 360
```

Use `uv sync --locked --group docs` when building the documentation. Add
runtime dependencies with `uv add`, development tools with `uv add --dev`, and
refresh all locked dependencies with `uv lock --upgrade`.

The pinned MOBI Fortran reference is a Git submodule. Initialize it after a
regular clone with `git submodule update --init`.

## Installed usage

```bash
$ pip install 'veros>=1.6.2' veros-bgc
$ veros copy-setup bgc_global_4deg --to /tmp/bgc-4deg
$ cd /tmp/bgc-4deg
$ veros run bgc_global_4deg.py
```

## Credits

Veros-BGC is based on [MOBI](http://people.oregonstate.edu/~schmita2/Models/MOBI/index.html) by [Andreas Schmittner](http://people.oregonstate.edu/~schmita2/), Oregon State University.

[Steffen Randrup](https://github.com/SteffenRandrup) created Veros-BGC as a part of his Master's thesis.
