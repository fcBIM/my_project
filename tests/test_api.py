
# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5

    # content of test_sysexit.py
import pytest


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()