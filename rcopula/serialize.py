r"""Save a fitted copula and load it back.

R's ``copula`` has no equivalent: an S4 object can go through ``saveRDS``, but
that is an opaque binary blob tied to the R session that wrote it. Here a copula
becomes a small, readable JSON document that can be committed to a repository,
sent over a wire, diffed in a pull request, or read by something that is not
Python at all.

The round trip is **exact**. Parameters go through Python's shortest
round-tripping float representation, which JSON preserves, so a reloaded copula
returns bit-identical densities -- not "close enough". That is the property
worth having: a model checked into version control today must price the same
book next year.

Structural constructions nest, because they nest in the objects too. A Khoudraji
copula whose second component is a rotated mixture of two Archimedeans serialises
to exactly that shape, and comes back as the same tree.

============================  ================================================
:func:`to_json`               Copula (or fit result) to a JSON string.
:func:`from_json`             ...and back.
:func:`to_dict`               The intermediate form, if JSON is not wanted.
:func:`from_dict`
============================  ================================================

Examples
--------
>>> import rcopula as rc
>>> from rcopula.serialize import from_json, to_json
>>> text = to_json(rc.ClaytonCopula(2.5, dim=3))
>>> reloaded = from_json(text)
>>> reloaded == rc.ClaytonCopula(2.5, dim=3)
True

Nested constructions survive intact:

>>> original = rc.KhoudrajiCopula(
...     rc.IndependenceCopula(2),
...     rc.RotatedCopula(rc.GumbelCopula(3.0), 180),
...     shapes=(0.4, 0.95),
... )
>>> from_json(to_json(original)).describe() == original.describe()
True

Notes
-----
A document records the version of ``rcopula`` that wrote it. Loading a document
from a *newer* version raises rather than guessing, because silently
mis-reading a model is worse than refusing to read it.

:class:`~rcopula.core.empirical.EmpiricalCopula` is deliberately not
serialisable: it *is* its data, and writing the data into the document would
turn a model file into a dataset -- with whatever licence and privacy
consequences that carries. Save the data yourself and rebuild it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rcopula.core.base import Copula

if TYPE_CHECKING:
    from rcopula.core.archimedean import ArchimedeanCopula
    from rcopula.structural.nested import NestedArchimedean

__all__ = ["from_dict", "from_json", "to_dict", "to_json"]

#: Bumped when a change to the document layout would make an old file
#: unreadable. Separate from the package version, which moves for many reasons
#: that have nothing to do with this format.
SCHEMA_VERSION = 2

# Schema 1 (rcopula 0.1.0) stored an unstructured elliptical copula's
# correlations in row-by-row lower-triangle order. Schema 2 uses R's
# column-by-column order. The two agree up to dim=3, so only dim >= 4 moves.
_ELLIPTICAL_KINDS = ("GaussianCopula", "StudentCopula")


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.atleast_1d(np.asarray(values, dtype=float))]


def _bools(values: Any) -> list[bool]:
    return [bool(v) for v in np.atleast_1d(np.asarray(values, dtype=bool))]


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def _encode(copula: Copula) -> dict[str, Any]:
    """One node of the tree. Structural copulas recurse; everything else is
    identified by class name and rebuilt from its parameters."""
    from rcopula.core.elliptical import EllipticalCopula, StudentCopula
    from rcopula.core.empirical import EmpiricalCopula
    from rcopula.structural.khoudraji import KhoudrajiCopula
    from rcopula.structural.mixture import MixtureCopula
    from rcopula.structural.nested import NestedArchimedean
    from rcopula.structural.opower import OuterPowerCopula
    from rcopula.structural.rotated import RotatedCopula
    from rcopula.vine import VineCopula

    kind = type(copula).__name__

    if isinstance(copula, EmpiricalCopula):
        raise TypeError(
            "an EmpiricalCopula is its data, and writing the data into a model "
            "document would turn it into a dataset -- with whatever licence and "
            "privacy that carries. Save the data separately and rebuild it."
        )

    if isinstance(copula, RotatedCopula):
        return {
            "kind": kind,
            "base": _encode(copula.base),
            "flip": _bools(copula.flip),
        }
    if isinstance(copula, OuterPowerCopula):
        return {
            "kind": kind,
            "base": _encode(copula.base),
            "alpha": float(copula.alpha),
        }
    if isinstance(copula, KhoudrajiCopula):
        return {
            "kind": kind,
            "copula1": _encode(copula.copula1),
            "copula2": _encode(copula.copula2),
            "shapes": _floats(copula.shapes),
        }
    if isinstance(copula, MixtureCopula):
        return {
            "kind": kind,
            "copulas": [_encode(component) for component in copula.copulas],
            "weights": _floats(copula.weights),
        }
    if isinstance(copula, NestedArchimedean):
        return {
            "kind": kind,
            "generator": _encode(copula.generator_copula),
            "components": [int(c) for c in copula.components],
            "children": [_encode(child) for child in copula.children],
        }
    if isinstance(copula, VineCopula):
        return {
            "kind": kind,
            "pair_copulas": [[_encode(pair) for pair in tree] for tree in copula.pair_copulas],
            "structure": str(copula.structure),
            "order": [int(i) for i in copula.order],
        }

    node: dict[str, Any] = {
        "kind": kind,
        "dim": int(copula.dim),
        "params": _floats(copula.params),
        "free": _bools(copula.free),
    }
    if isinstance(copula, EllipticalCopula):
        node["dispstr"] = str(copula.dispstr)
    if isinstance(copula, StudentCopula):
        # df lives in `params` for a free-df copula and outside it otherwise, so
        # record the flag rather than trying to infer it on the way back.
        node["df_fixed"] = bool(getattr(copula, "df_fixed", False))
    return node


def to_dict(copula: Copula) -> dict[str, Any]:
    """Convert a copula into a plain Python dictionary you can store or send anywhere.

    The dictionary holds only strings, numbers, booleans, lists and nested
    dictionaries, so it can go into JSON, YAML or a database as-is. Use
    :func:`to_json` if you just want a JSON string, and :func:`from_dict` to
    turn the dictionary back into a copula.

    Parameters
    ----------
    copula : Copula
        The copula to save: any parametric family, or a structural
        construction (rotated, outer power, Khoudraji, mixture, nested
        Archimedean, vine) built from them. Not an
        :class:`~rcopula.core.empirical.EmpiricalCopula`.

    Returns
    -------
    dict of str to Any
        ``"rcopula"`` : str
            Version of rcopula that wrote the document.
        ``"schema"`` : int
            Layout version of the document (currently 2).
        ``"copula"`` : dict
            The copula itself. Every node has a ``"kind"`` (the class name).
            A plain family also has ``"dim"`` (int), ``"params"`` (list of
            float) and ``"free"`` (list of bool), plus ``"dispstr"`` for the
            elliptical families and ``"df_fixed"`` for the Student t. A
            structural copula instead holds its components as nested nodes.

    Raises
    ------
    TypeError
        If the copula is (or contains) an
        :class:`~rcopula.core.empirical.EmpiricalCopula`, which is
        deliberately not serialisable.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.serialize import to_dict
    >>> document = to_dict(rc.GumbelCopula(2.0))
    >>> document["copula"]["kind"], document["copula"]["params"]
    ('GumbelCopula', [2.0])
    """
    # Imported here rather than at module scope: rcopula/__init__ imports this
    # module, so a top-level import would be circular.
    from rcopula import __version__

    return {
        "rcopula": __version__,
        "schema": SCHEMA_VERSION,
        "copula": _encode(copula),
    }


def to_json(copula: Copula, *, indent: int | None = 2) -> str:
    """Save a copula as a JSON text string, so it can be written to a file and reloaded exactly.

    The string is the JSON form of :func:`to_dict`'s output. Read it back
    with :func:`from_json`; the reloaded copula is bit-identical.

    Parameters
    ----------
    copula : Copula
        The copula to save; see :func:`to_dict` for what is accepted.
    indent : int or None, default 2
        Spaces of indentation, passed to :func:`json.dumps`. ``None`` gives
        the compact single-line form.

    Returns
    -------
    str
        The JSON document.

    Raises
    ------
    TypeError
        If the copula is (or contains) an
        :class:`~rcopula.core.empirical.EmpiricalCopula`.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.serialize import to_json
    >>> print(to_json(rc.FrankCopula(4.0), indent=None))
    {"rcopula": ..., "schema": 2, "copula": {"kind": "FrankCopula", ...}}
    """
    return json.dumps(to_dict(copula), indent=indent)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def _families() -> dict[str, type[Copula]]:
    """Class name to class, for everything reconstructible from parameters."""
    import rcopula as rc

    names = [
        "AMHCopula",
        "ClaytonCopula",
        "FGMCopula",
        "FrankCopula",
        "FrechetLowerCopula",
        "FrechetUpperCopula",
        "GalambosCopula",
        "GaussianCopula",
        "GumbelCopula",
        "HuslerReissCopula",
        "IndependenceCopula",
        "JoeCopula",
        "MarshallOlkinCopula",
        "PlackettCopula",
        "StudentCopula",
        "TawnCopula",
        "TEVCopula",
    ]
    return {name: getattr(rc, name) for name in names}


def _migrate_v1(node: Any) -> Any:
    """Rewrite a schema-1 tree's unstructured correlations into schema-2 order."""
    if isinstance(node, list):
        return [_migrate_v1(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _migrate_v1(value) for key, value in node.items()}
    dim = int(out.get("dim", 0) or 0)
    if out.get("kind") in _ELLIPTICAL_KINDS and out.get("dispstr") == "un" and dim >= 4:
        n = dim * (dim - 1) // 2
        rows, cols = np.tril_indices(dim, -1)
        old = dict(
            zip(zip(rows.tolist(), cols.tolist(), strict=True), out["params"][:n], strict=True)
        )
        cols2, rows2 = np.triu_indices(dim, 1)
        new = [old[(r, c)] for r, c in zip(rows2.tolist(), cols2.tolist(), strict=True)]
        out["params"] = new + list(out["params"][n:])
        if "free" in out and len(out["free"]) >= n:
            free_old = dict(
                zip(zip(rows.tolist(), cols.tolist(), strict=True), out["free"][:n], strict=True)
            )
            free_new = [
                free_old[(r, c)] for r, c in zip(rows2.tolist(), cols2.tolist(), strict=True)
            ]
            out["free"] = free_new + list(out["free"][n:])
    return out


def _decode(node: dict[str, Any]) -> Copula:
    import rcopula as rc
    from rcopula.core.elliptical import EllipticalCopula, StudentCopula

    kind = node.get("kind")
    if not isinstance(kind, str):
        raise ValueError(f"a copula node needs a string 'kind'; got {kind!r}")

    if kind == "RotatedCopula":
        return rc.RotatedCopula(_decode(node["base"]), np.asarray(node["flip"], dtype=bool))
    if kind == "OuterPowerCopula":
        base = _decode(node["base"])
        return rc.OuterPowerCopula(cast("ArchimedeanCopula", base), float(node["alpha"]))
    if kind == "KhoudrajiCopula":
        return rc.KhoudrajiCopula(
            _decode(node["copula1"]), _decode(node["copula2"]), shapes=node["shapes"]
        )
    if kind == "MixtureCopula":
        return rc.MixtureCopula(
            [_decode(component) for component in node["copulas"]], weights=node["weights"]
        )
    if kind == "NestedArchimedean":
        children = [_decode(child) for child in node["children"]]
        return rc.NestedArchimedean(
            _decode(node["generator"]),
            components=node["components"],
            children=cast("list[NestedArchimedean]", children),
        )
    if kind == "VineCopula":
        return rc.VineCopula(
            [[_decode(pair) for pair in tree] for tree in node["pair_copulas"]],
            structure=node["structure"],
            order=node["order"],
        )

    families = _families()
    if kind not in families:
        raise ValueError(
            f"unknown copula {kind!r}; this document was probably written by a "
            f"different version of rcopula. Known: {sorted(families)}"
        )
    cls = families[kind]
    dim = int(node["dim"])
    params = np.asarray(node["params"], dtype=float)
    free = np.asarray(node["free"], dtype=bool)

    kwargs: dict[str, Any] = {"dim": dim}
    if issubclass(cls, EllipticalCopula) and "dispstr" in node:
        kwargs["dispstr"] = node["dispstr"]
    if issubclass(cls, StudentCopula):
        # The degrees of freedom sit at the end of `params` either way; the flag
        # only decides whether they are also a free parameter.
        kwargs["df"] = float(params[-1])
        kwargs["df_fixed"] = bool(node.get("df_fixed", False))

    # Build the copula unparameterised and then set the parameters, rather than
    # passing them to the constructor: the one-parameter Archimedeans take a
    # *scalar* theta while the elliptical families take a vector, and going
    # through with_params sidesteps that difference entirely.
    builder = cast("Any", cls)
    try:
        copula = builder(dim=dim, **{k: v for k, v in kwargs.items() if k != "dim"})
    except TypeError:
        copula = builder(dim=dim)
    if params.size:
        copula = copula.with_params(params)
    if free.size and free.size == np.asarray(copula.free).size and not free.all():
        copula = copula.fix_params(free)
    return copula


def from_dict(document: dict[str, Any]) -> Copula:
    """Turn a dictionary made by :func:`to_dict` back into a working copula.

    Documents written by older versions of rcopula (schema 1) are upgraded
    automatically on the way in.

    Parameters
    ----------
    document : dict of str to Any
        A dictionary as returned by :func:`to_dict`, with at least the keys
        ``"schema"`` and ``"copula"``.

    Returns
    -------
    Copula
        The rebuilt copula, of the same class and with the same parameters
        (and fixed/free flags) as the one that was saved.

    Raises
    ------
    ValueError
        If the dictionary has no ``"schema"`` or ``"copula"`` field, was
        written by a newer schema, or names a copula this version does not
        know. Refusing is deliberate: silently mis-reading a model is worse
        than not reading it.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.serialize import from_dict, to_dict
    >>> from_dict(to_dict(rc.JoeCopula(3.0))) == rc.JoeCopula(3.0)
    True
    """
    schema = document.get("schema")
    if schema is None:
        raise ValueError("not an rcopula document: no 'schema' field")
    if int(schema) > SCHEMA_VERSION:
        from rcopula import __version__

        raise ValueError(
            f"this document uses schema {schema}, and this rcopula "
            f"({__version__}) understands up to {SCHEMA_VERSION}. Upgrade "
            "rcopula rather than reading it approximately."
        )
    if "copula" not in document:
        raise ValueError("not an rcopula document: no 'copula' field")
    node = document["copula"]
    if int(schema) < 2:
        node = _migrate_v1(node)
    return _decode(node)


def from_json(text: str) -> Copula:
    """Load a copula back from a JSON string made by :func:`to_json`.

    Parameters
    ----------
    text : str
        The JSON document, for example the contents of a file written from
        :func:`to_json`.

    Returns
    -------
    Copula
        The rebuilt copula.

    Raises
    ------
    json.JSONDecodeError
        If ``text`` is not valid JSON.
    ValueError
        For the same reasons as :func:`from_dict`.

    Examples
    --------
    The round trip is exact, not approximate -- the reloaded copula gives
    bit-identical densities:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.serialize import from_json, to_json
    >>> original = rc.ClaytonCopula(2.3456789012345678, dim=4)
    >>> reloaded = from_json(to_json(original))
    >>> u = original.rvs(50, random_state=0)
    >>> bool(np.array_equal(original.logpdf(u), reloaded.logpdf(u)))
    True
    """
    return from_dict(json.loads(text))
