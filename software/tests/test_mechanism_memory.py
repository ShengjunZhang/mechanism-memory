import unittest

from causal_sandbox.mechanism_memory import IdentityMemory, MechanismTuple


class IdentityMemoryTests(unittest.TestCase):
    def test_edit_removes_old_mutable_key_and_preserves_untouched_identity(self):
        old = MechanismTuple("a", "c0", "y0", 0, 1)
        stable = MechanismTuple("b", "c0", "z", 1, 1)
        new = MechanismTuple("a", "c1", "y1", 1, 3)
        memory = IdentityMemory({1: old, 2: stable})
        memory.local_update({1: new})
        self.assertEqual(memory.tuples(), {new, stable})
        self.assertNotIn(old.lookup_key, memory.index)
        self.assertEqual(memory.index[new.lookup_key], {1})
        self.assertIs(memory.entries[2], stable)

    def test_shared_key_keeps_other_identity_when_one_owner_moves(self):
        first = MechanismTuple("a", "c", "y", 0, 1)
        second = MechanismTuple("a", "c", "y", 1, 1)
        memory = IdentityMemory({1: first, 2: second})
        replacement = MechanismTuple("a", "c", "z", 1, 2)
        memory.replace(1, replacement)
        self.assertEqual(memory.index[first.lookup_key], {2})
        self.assertEqual(memory.tuples(), {second, replacement})

    def test_unknown_local_identity_fails_before_any_partial_update(self):
        value = MechanismTuple("a", "c", "y", 0, 1)
        memory = IdentityMemory({1: value})
        with self.assertRaises(KeyError):
            memory.local_update({1: value._replace(value=1), 3: value})
        self.assertEqual(memory.entries, {1: value})


if __name__ == "__main__":
    unittest.main()
