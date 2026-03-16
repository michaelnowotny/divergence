from divergence import intersection


def test_non_overlapping_1():
    assert intersection(1, 2, 3, 4) is None


def test_non_overlapping_2():
    assert intersection(3, 4, 1, 2) is None


def test_sub_interval():
    assert intersection(1, 4, 2, 3) == (2, 3)


def test_overlap():
    assert intersection(2, 4, 3, 5) == (3, 4)


def test_sub_overlap_2():
    assert intersection(3, 5, 2, 4) == (3, 4)
