import os
import sys

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

from up_cpor.converter import UpCporConverter  # initializes the CLR
from CPORLib.LogicalUtilities import GroundedPredicate, Predicate
from CPORLib.Tools import GenericArraySet

from cpor_test_utils import TEST_RANDOM_SEED, reset_test_seeds


def setup_function():
    GenericArraySet[Predicate].Reset()


def teardown_function():
    GenericArraySet[Predicate].Reset()


def test_reset_clears_grounded_predicate_index_fields():
    """Regression: Reset() must set gp.Index = -1 on all tracked GroundedPredicates.

    Before the fix, Reset() replaced the Indexes dict and zeroed CountIndexes but
    left gp.Index unchanged.  After Reset() the GetIndex() fast-path
    ``if (gp.Index != -1) return gp.Index;`` returned the stale index without
    registering the predicate in the new Indexes dict, so Indexes stayed empty
    while CountIndexes stayed at 0, producing index collisions.
    """
    p = GroundedPredicate("p")
    q = GroundedPredicate("q")
    s = GenericArraySet[Predicate]()
    s.Add(p)
    s.Add(q)
    assert p.Index != -1  # sanity: indices were assigned
    assert q.Index != -1

    GenericArraySet[Predicate].Reset()

    assert p.Index == -1, f"p.Index should be -1 after Reset(), got {p.Index}"
    assert q.Index == -1, f"q.Index should be -1 after Reset(), got {q.Index}"


def test_reset_prevents_index_collision_on_reuse_after_reset():
    """Regression: stale gp.Index causes index collisions when predicates are reused after Reset().

    Without the fix: after Reset(), stale predicates bypass Indexes registration
    (gp.Index != -1 fast-paths them), leaving Indexes empty while CountIndexes=0.
    The first truly-fresh predicate is then assigned index 0, colliding with the
    stale predicate that also holds index 0.

    With the fix: Reset() clears gp.Index to -1 for all tracked predicates, so
    every Add() after Reset() goes through the slow path and registers in Indexes
    with a fresh, contiguous index.
    """
    p = GroundedPredicate("p")
    q = GroundedPredicate("q")
    r = GroundedPredicate("r")  # fresh — not seen before the reset

    first = GenericArraySet[Predicate]()
    first.Add(p)
    first.Add(q)
    # p.Index == 0, q.Index == 1 at this point
    GenericArraySet[Predicate].Reset()

    second = GenericArraySet[Predicate]()
    second.Add(p)
    second.Add(q)
    second.Add(r)

    indices = {p.Index, q.Index, r.Index}
    assert len(indices) == 3, (
        f"Expected 3 distinct indices after Reset()+re-add; got collision: "
        f"p={p.Index}, q={q.Index}, r={r.Index}"
    )

    # Every predicate must be registered in the new Indexes dict.
    assert p in GenericArraySet[Predicate].Indexes, "p missing from Indexes after Reset()+Add()"
    assert q in GenericArraySet[Predicate].Indexes, "q missing from Indexes after Reset()+Add()"
    assert r in GenericArraySet[Predicate].Indexes, "r missing from Indexes after Reset()+Add()"


def test_reset_test_seeds_clears_grounded_predicate_index_fields():
    """Regression: reset_test_seeds() must propagate the C# static-state reset.

    Before the fix, reset_test_seeds() did not call GenericArraySet[Predicate].Reset(),
    so gp.Index values on surviving GroundedPredicate objects were never cleared
    between tests.  This caused the index-collision bug described in
    test_reset_prevents_index_collision_on_reuse_after_reset to appear between
    tests even when each test called reset_test_seeds() at its start.
    """
    p = GroundedPredicate("p")
    q = GroundedPredicate("q")
    s = GenericArraySet[Predicate]()
    s.Add(p)
    s.Add(q)
    assert p.Index != -1
    assert q.Index != -1

    reset_test_seeds(TEST_RANDOM_SEED)

    assert p.Index == -1, (
        f"reset_test_seeds() must clear gp.Index; p.Index={p.Index} after reset"
    )
    assert q.Index == -1, (
        f"reset_test_seeds() must clear gp.Index; q.Index={q.Index} after reset"
    )
