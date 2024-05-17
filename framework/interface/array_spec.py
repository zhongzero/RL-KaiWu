#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pydoc import locate
import numpy as np


class ArraySpec:
    """
    用来描述state和action中numpy数组的规范, 包括shape和dtype, 以及抽样分布类
    """

    def __init__(self, shape, dtype, pdclass='framework.common.algo.distribution.CategoricalDist'):
        self._shape = shape
        self._dtype = dtype
        self._pdclass = locate(pdclass)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def pdclass(self):
        return self._pdclass

    def __str__(self):
        return f"{str(self._shape), str(self._dtype)}"

    def __repr__(self):
        return f"ArraySpec({self._shape})"


class Box(ArraySpec):
    """
        A (possibly unbounded) box in R^n. Specifically, a Box represents the
        Cartesian product of n closed intervals. Each interval has the form of one
        of [a, b], (-oo, b], [a, oo), or (-oo, oo).

        There are two common use cases:

        * Identical bound for each dimension::
            >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
            Box(3, 4)

        * Independent bound for each dimension::
            >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
            Box(2,)
        """

    def __init__(self, low, high, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self._dtype = np.dtype(dtype)

        if shape is None:
            assert low.shape == high.shape, 'box dimension mismatch. '
            self._shape = low.shape
            self._low = low
            self._high = high
        else:
            assert np.isscalar(low) and np.isscalar(high), 'box requires scalar bounds. '
            self._shape = tuple(shape)
            self._low = np.full(self._shape, low)
            self._high = np.full(self._shape, high)

        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            return np.inf

        low_precision = _get_precision(self._low.dtype)
        high_precision = _get_precision(self._high.dtype)
        dtype_precision = _get_precision(self._dtype)
        assert min(low_precision, high_precision) >= dtype_precision, \
            "Box bound precision lowered by casting to {}".format(self._dtype)

        self._low = self._low.astype(self._dtype)
        self._high = self._high.astype(self._dtype)

        super(Box, self).__init__(self._shape, self._dtype)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def __repr__(self):
        return f"Box({self._shape})"


class Discrete(ArraySpec):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
        Example::
            >>> Discrete(2)
        """

    def __init__(self, n, shape=()):
        assert n >= 0
        self._n = n
        self._shape = shape
        super(Discrete, self).__init__(self._shape, np.int64)

    @property
    def n(self):
        return self._n

    def __repr__(self):
        return f"Discrete({self._shape})(n_class={self._n})"


class MultiDiscrete(ArraySpec):
    """
        - The multi-discrete action space consists of a series of discrete action spaces with different
        number of actions in eachs
        - It is useful to represent game controllers or keyboards where each key can be represented as
        a discrete action space
        - It is parametrized by passing an array of positive integers specifying number of actions
        for each discrete action space
        Note: Some environment wrappers assume a value of 0 always represents the NOOP action.
        e.g. Nintendo Game Controller
        - Can be conceptualized as 3 discrete action spaces:
            1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
            2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
            3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        - Can be initialized as
            MultiDiscrete([ 5, 2, 2 ])
    """

    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (np.array(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self._nvec = np.asarray(nvec, dtype=np.int64)

        super(MultiDiscrete, self).__init__(self._nvec.shape, np.int64)

    @property
    def nvec(self):
        return self._nvec

    def __repr__(self):
        return f"MultiDiscrete({self._nvec.shape})"
