# Encoding: utf-8
"""Constant refractive index."""

import numpy.typing as npt

from .base_dispersion import IndexDispersion


class ConstantRefractiveIndex(IndexDispersion):
    r"""Constant refractive index.

    Single parameters:
        :n: The constant value of the refractive index. Defaults to 1.

    Repeated parameters:
        --

    Output:
        .. math::
            \boldsymbol{n}
    """

    single_params_template = {"n": 1}
    rep_params_template = {}

    def refractive_index(self, _: npt.ArrayLike) -> npt.NDArray:
        return self.single_params.get("n")
