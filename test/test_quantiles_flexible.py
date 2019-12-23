import pytest

import numpy as np

from src.quantiles_flexible import KLLR


def test_top_min_capacity():
    for mem in range(6, 30):
        kll = KLLR(mem)
        k = kll._top_min_capacity
        s = 0
        while k >= kll._initial_min_compactor_size:
            s += k
            k = k // 3 * 2
        assert s <= mem
        assert s + kll._top_min_capacity * 3 // 2 + (kll._top_min_capacity * 3 // 2) % 2 > mem


def test_update():
    # TODO - more to add here..
    n_batch = 3
    batch = 20

    kll = KLLR(100)
    for i in range(n_batch):
        data = np.random.uniform(low=-1, high=1, size=(batch, 1))
        kll.update(data)


def test_l0_capacity():
    mem = 200
    step = 21

    kll = KLLR(mem)
    assert kll._l0_capacity() == mem
    for i in range(mem // step):
        kll.update(np.random.uniform(size=(step, 1)))
        assert kll._l0_capacity() == mem - (i+1)*step, f'{kll._l0_capacity()} != {mem - (i+1)*step}'


