# Encoding: utf-8
"""Cauchy dispersion."""

import numpy.typing as npt

from .base_dispersion import IndexDispersion


class Cauchy(IndexDispersion):
    r"""Cauchy dispersion.

    Single parameters:
        :n0: Defaults to 1.5.
        :n1: Defaults to 0. Unit in nm\ :sup:`2`.
        :n2: Defaults to 0. Unit in nm\ :sup:`4`.
        :k0: Defaults to 0.
        :k1: Defaults to 0. Unit in nm\ :sup:`2`.
        :k2: Defaults to 0. Unit in nm\ :sup:`4`.

    Repeated parameters:
        --

    Output:
        .. math::
            \varepsilon^{1/2}(\lambda) = n(\lambda) =
            \boldsymbol{n_0} + 100  \boldsymbol{n_1}/\lambda^2 + 10^7 \boldsymbol{n_2}/\lambda^4
            + i (\boldsymbol{k_0} + 100 \boldsymbol{k_1}/\lambda^2
            + 10^7 \boldsymbol{k_2}/\lambda^4)
    """

    single_params_template = {"n0": 1.5, "n1": 0, "n2": 0, "k0": 0, "k1": 0, "k2": 0}
    rep_params_template = {}

    def refractive_index(self, lbda: npt.ArrayLike) -> npt.NDArray:
        return (
            self.single_params.get("n0")
            + 1e2 * self.single_params.get("n1") / lbda**2
            + 1e7 * self.single_params.get("n2") / lbda**4
            + 1j
            * (
                self.single_params.get("k0")
                + 1e2 * self.single_params.get("k1") / lbda**2
                + 1e7 * self.single_params.get("k2") / lbda**4
            )
        )
