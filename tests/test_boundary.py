"""DISCO-CMB uses DISCO-EB through its interface, the :mod:`discoeb` package.

Every module of :mod:`discocmb` may import the names :mod:`discoeb` exports
and the generic numerics of :mod:`discoeb.spline_interpolation`, nothing else
of DISCO-EB.
Checked on the import statements, so a private reach across the boundary --
``from discoeb.perturbations import _launch`` -- fails here rather than in
review.
"""

import ast
from pathlib import Path

import discocmb

PACKAGE = Path(discocmb.__file__).parent
ALLOWED = {"discoeb", "discoeb.spline_interpolation"}


def _discoeb_imports(path: Path) -> set[str]:
    """The ``discoeb`` modules a file imports, at any depth of its code."""

    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "discoeb":
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(
                a.name for a in node.names if a.name.split(".")[0] == "discoeb"
            )
    return found


def test_discocmb_uses_the_einstein_boltzmann_interface_only():
    for path in sorted(PACKAGE.glob("*.py")):
        outside = _discoeb_imports(path) - ALLOWED
        assert not outside, f"discocmb.{path.stem} imports {sorted(outside)}"


def test_what_discocmb_uses_is_exported():
    import discoeb

    names = set()
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "discoeb":
                names.update(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "eb"
            ):
                names.add(node.attr)
    exported = set(discoeb.__all__)
    assert names <= exported, sorted(names - exported)
