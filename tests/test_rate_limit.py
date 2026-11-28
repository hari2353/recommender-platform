from recsys.rate_limit import TokenBucket


def test_allows_up_to_capacity():
    b = TokenBucket(3, 1)
    assert b.allow(now=0) is True
    assert b.allow(now=0) is True
    assert b.allow(now=0) is True
    assert b.allow(now=0) is False


def test_refills_over_time():
    b = TokenBucket(2, 1)
    b.allow(now=0)
    b.allow(now=0)
    assert b.allow(now=0) is False
    assert b.allow(now=2) is True


def test_invalid():
    try:
        TokenBucket(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("zero capacity accepted")
    b = TokenBucket(2, 1)
    try:
        b.allow(0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero n accepted")
    try:
        b.allow(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative n accepted")
