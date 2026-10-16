import pytest

from specwire.verify import greedy_accept


def test_all_guesses_accepted():
    assert greedy_accept([5, 9, 2, 7], [5, 9, 2, 7, 8]) == (4, 8)


def test_first_guess_wrong():
    assert greedy_accept([5, 9], [4, 9, 1]) == (0, 4)


def test_mismatch_in_middle():
    assert greedy_accept([5, 9, 2, 7], [5, 9, 3, 1, 8]) == (2, 3)


def test_empty_draft():
    assert greedy_accept([], [6]) == (0, 6)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        greedy_accept([5, 9], [5, 9])
