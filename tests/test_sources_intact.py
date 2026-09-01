"""Every source file must be a single, valid, non-duplicated module.

serve.py was silently written twice, end to end, on two separate occasions.
Python does not complain about most duplicated modules - the second set of
definitions quietly replaces the first - so it ran fine and nothing noticed.
serve.py only failed loudly because it happens to start with a `from __future__`
import, which is illegal anywhere but the top of a file.

Any other file would have gone on working while carrying a hidden second copy
of itself, and the next edit would have landed in one half only. So the check
is on every file rather than the one that broke.
"""

from pathlib import Path

import pytest

# Tests are included deliberately. The first version of this guard covered
# only app/, tools/ and the entrypoints - and while it was in place, two files
# under tests/ were silently duplicated and nobody noticed, because a duplicated
# test file still passes: pytest simply collects each test twice. A guard that
# does not cover the place a fault actually occurred is not a guard.
SOURCES = sorted(
    p for p in
    list(Path("app").rglob("*.py")) + list(Path("tools").rglob("*.py"))
    + list(Path("tests").rglob("*.py")) + [Path("main.py"), Path("serve.py")]
    if p.is_file()
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p))
def test_a_source_file_is_not_a_copy_of_itself(path):
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if len(lines) < 4:
        return

    # A duplicated file is exactly two identical halves.
    middle = len(lines) // 2
    first = "\n".join(lines[:middle]).strip()
    second = "\n".join(lines[middle:]).strip()
    assert not (first and first == second), (
        f"{path} appears to contain two identical copies of itself"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p))
def test_a_source_file_declares_its_module_docstring_once(path):
    """A second module docstring at the halfway mark is the duplication tell."""
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith('"""'):
        return
    opening = text.split("\n", 1)[0]
    if len(opening) < 8:                       # a one-word docstring is not distinctive
        return
    assert text.count(opening) == 1, (
        f"{path} repeats its opening line {text.count(opening)} times - "
        f"the file may have been written twice"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p))
def test_a_source_file_compiles(path):
    import ast

    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"{path} does not parse: {exc.msg} at line {exc.lineno}")


def test_no_test_function_is_defined_twice_in_one_file():
    """A duplicated test file passes silently - pytest just runs each test twice.

    That is exactly how two files here stayed duplicated while the guard above
    was already in place: nothing failed, the count merely inflated. A repeated
    function name catches it even when the duplication is partial rather than a
    clean doubling of the whole file.
    """
    import re

    repeats = {}
    for path in Path("tests").rglob("*.py"):
        names = re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.M)
        duplicated = {n for n in names if names.count(n) > 1}
        if duplicated:
            repeats[str(path)] = sorted(duplicated)

    assert not repeats, f"test functions defined more than once: {repeats}"
