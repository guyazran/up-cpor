import pytest

from cpor_test_utils import TEST_RANDOM_SEED, reset_test_seeds, sanitize_sys_path_for_pythonnet

sanitize_sys_path_for_pythonnet()


@pytest.fixture(autouse=True)
def _reset_test_rng_state():
    reset_test_seeds(TEST_RANDOM_SEED)
