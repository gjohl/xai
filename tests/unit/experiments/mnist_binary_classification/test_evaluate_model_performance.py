import pytest

from xai.experiments.helpers import get_count_per_digit


@pytest.mark.parametrize('digits, num_samples_per_class, out_of_dist_pct, expected', [
    [(0, 1, 2), 100, 0., {0: 100, 1: 100, 2: 0}],
    [(0, 1, 2), 100, 0.3, {0: 70, 1: 100, 2: 30}],
    [(0, 1, 2), 100, 1., {0: 0, 1: 100, 2: 100}],
    [(0, 1, 2), 1000, 0.5, {0: 500, 1: 1000, 2: 500}],
    [(0, 1, 2, 3), 100, 0.4, {0: 60, 1: 100, 2: 20, 3: 20}],
    [(0, 1, 2, 3, 4, 5), 100, 0.4, {0: 60, 1: 100, 2: 10, 3: 10, 4: 10, 5: 10}],
])
def test_get_count_per_digit(digits, num_samples_per_class, out_of_dist_pct, expected):
    actual = get_count_per_digit(digits, num_samples_per_class, out_of_dist_pct)
    assert actual == expected
