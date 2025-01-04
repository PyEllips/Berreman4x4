# Encoding: utf-8
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.constants as sc

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

from numpy.lib.scimath import sqrt
from scipy.linalg import expm as scipy_expm

from .materials import IsotropicMaterial
from .result import Result
from .solver import Solver


class Propagator(ABC):
    """Propagator abstract base class."""

    @abstractmethod
    def calculate_propagation(
        self, delta: npt.NDArray, thickness: float, lbda: npt.ArrayLike
    ) -> npt.NDArray:
        """Calculates propagation for a given Delta matrix and layer thickness.

        Args:
            delta (npt.NDArray): Delta Matrix
            thickness (float): Thickness of layer (nm)
            lbda (npt.ArrayLike): Wavelengths to evaluate (nm)

        Returns:
            npt.NDArray: Propagator for the given layer
        """


class PropagatorLinear(Propagator):
    """Propagator class using a simple linear approximation of the matrix exponential."""

    def calculate_propagation(
        self, delta: npt.NDArray, thickness: float, lbda: npt.ArrayLike
    ) -> npt.NDArray:
        """Calculates propagation for a given Delta matrix and layer thickness with a linear approximation of the matrix exponential.

        Args:
            delta (npt.NDArray): Delta Matrix
            thickness (float): Thickness of layer (nm)
            lbda (npt.ArrayLike): Wavelengths to evaluate (nm)

        Returns:
            npt.NDArray: Propagator for the given layer
        """
        p_hs_lin = np.identity(4) + 1j * thickness * np.einsum(
            "nij,n->nij", delta, 2 * sc.pi / lbda
        )
        return p_hs_lin


class PropagatorExpm(Propagator):
    """Propagator class using the Padé approximation of the matrix exponential."""

    def __init__(self, backend: Literal["torch", "scipy", "automatic"] = "automatic"):
        """The Propagator can use two different backends: SciPy and PyTorch.
        The default installation only provides SciPy.
        PyTorch is faster and will be used automatically if available.
        If you want to install PyTorch please follow the instructions at https://pytorch.org/get-started/locally/.

        Args:
            backend (Literal["torch", "scipy", "automatic"], optional): Setting to change the linear algebra provider. Defaults to "automatic".
        """
        backends = {
            "torch": lambda mats: torch.linalg.matrix_exp(
                torch.from_numpy(mats)
            ).numpy(),
            "scipy": lambda mats: scipy_expm(mats),
        }

        if backend == "automatic" and TORCH_AVAILABLE:
            backend = "torch"
        elif backend == "automatic" and not TORCH_AVAILABLE:
            backend = "scipy"
        elif backend == "torch" and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. If you want to use the PyTorch backend, \
            please follow the install instructions on https://pytorch.org/get-started/locally/"
            )
        elif backend not in backends:
            raise ValueError(
                "Backend should be one of 'torch', 'scipy' or 'automatic'."
            )

        self.expm = backends[backend]

    def calculate_propagation(
        self, delta: npt.NDArray, thickness: float, lbda: npt.ArrayLike
    ) -> npt.NDArray:
        """Calculates propagation for a given Delta matrix and layer thickness with the Padé approximation of the matrix exponential.

        Args:
            delta (npt.NDArray): Delta Matrix
            thickness (float): Thickness of layer (nm)
            lbda (npt.ArrayLike): Wavelengths to evaluate (nm)

        Returns:
            npt.NDArray: Propagator for the given layer
        """
        mats = 1j * thickness * np.einsum("nij,n->nij", delta, 2 * sc.pi / lbda)

        propagator = self.expm(mats)

        return propagator


class PropagatorEig(Propagator):
    """Propagator class using the eigenvalue decomposition method."""

    def calculate_propagation(
        self, delta: npt.NDArray, thickness: float, lbda: npt.ArrayLike
    ) -> npt.NDArray:
        """Calculates propagation for a given Delta matrix and layer thickness with eigenvalue decomposition.

        Args:
            delta (npt.NDArray): Delta Matrix
            thickness (float): Thickness of layer (nm)
            lbda (npt.ArrayLike): Wavelengths to evaluate (nm)

        Returns:
            npt.NDArray: Propagator for the given layer
        """
        q, w = np.linalg.eig(delta)

        # Sort according to z propagation direction, by Re(q) first, then Im(q)
        i = np.lexsort((-np.real(q), -np.imag(q)))

        q = np.take_along_axis(q, i, axis=-1)
        w = np.take_along_axis(w, i[:, np.newaxis, :], axis=-1)

        w_i = np.linalg.inv(w)

        q = np.exp(q * 2j * thickness * sc.pi / lbda[:, None])

        p = np.zeros((lbda.shape[0], 4, 4), dtype=np.complex128)
        for i in range(4):
            p[:, i, i] = q[:, i]

        return w @ p @ w_i


class Solver4x4(Solver):
    """Solver class to evaluate Experiment objects. Based on Berreman's 4x4 method."""

    @staticmethod
    def build_delta_matrix(k_x: npt.ArrayLike, eps: npt.NDArray) -> npt.NDArray:
        """Calculates Delta matrix for given permittivity and reduced wave number.

        Args:
            k_x (npt.ArrayLike): reduce wave number, Kx = kx/k0
            eps (npt.NDArray): permittivity tensor

        Returns:
            npt.NDArray: Delta 4x4 matrix: infinitesimal propagation matrix
        """
        if np.shape(k_x) == ():
            length = 1
        else:
            length = np.shape(k_x)[0]

        zeros = np.tile(0, length)
        ones = np.tile(1, length)

        delta = np.array(
            [
                [
                    -k_x * eps[:, 2, 0] / eps[:, 2, 2],
                    -k_x * eps[:, 2, 1] / eps[:, 2, 2],
                    zeros,
                    ones - k_x**2 / eps[:, 2, 2],
                ],
                [zeros, zeros, -ones, zeros],
                [
                    eps[:, 1, 2] * eps[:, 2, 0] / eps[:, 2, 2] - eps[:, 1, 0],
                    k_x**2 - eps[:, 1, 1] + eps[:, 1, 2] * eps[:, 2, 1] / eps[:, 2, 2],
                    zeros,
                    k_x * eps[:, 1, 2] / eps[:, 2, 2],
                ],
                [
                    eps[:, 0, 0] - eps[:, 0, 2] * eps[:, 2, 0] / eps[:, 2, 2],
                    eps[:, 0, 1] - eps[:, 0, 2] * eps[:, 2, 1] / eps[:, 2, 2],
                    zeros,
                    -k_x * eps[:, 0, 2] / eps[:, 2, 2],
                ],
            ],
            dtype=np.complex128,
        )
        delta = np.moveaxis(delta, 2, 0)
        return delta

    @staticmethod
    def transition_matrix_halfspace(delta: npt.NDArray) -> npt.NDArray:
        """Returns transition exit matrix L for any half-space.

        Sort eigenvectors of the Delta matrix according to propagation
        direction first, then according to $y$ component.

        Returns eigenvectors ordered like (s+,s-,p+,p-)

        Args:
            delta (npt.NDArray): Delta 4x4 matrix: infinitesimal propagation matrix

        Returns:
            npt.NDArray: Translation matrix for semi-infinite half-spaces
        """
        q, p = np.linalg.eig(delta)

        # Sort according to z propagation direction, by Re(q) first, then Im(q)
        idx = np.lexsort((-np.real(q), -np.imag(q)))

        q = np.take_along_axis(q, idx, axis=-1)
        p = np.take_along_axis(p, idx[:, np.newaxis, :], axis=-1)
        # Result should be (+,+,-,-)

        # For each direction, sort according to Ey component, highest Ey first
        i1 = np.argsort(-np.abs(p[:, 1, :2]))
        i2 = 2 + np.argsort(-np.abs(p[:, 1, 2:]))
        i = np.hstack((i1, i2))
        # Result should be (s+,p+,s-,p-)

        # Reorder
        i[:, [1, 2]] = i[:, [2, 1]]

        q = np.take_along_axis(q, i, axis=-1)
        p = np.take_along_axis(p, i[:, np.newaxis, :], axis=-1)
        # Result should be(s+,s-,p+,p-)

        # Adjust Ey in ℝ⁺ for 's', and Ex in ℝ⁺ for 'p'
        e = np.hstack((p[:, 1, :2], p[:, 0, 2:]))

        ne = np.abs(e)
        c = np.ones_like(e)
        i = ne != 0.0
        c[i] = e[i] / ne[i]

        p = p * c[:, np.newaxis, :]

        # Normalize so that Ey = c1 + c2, analog to Ey = Eis + Ers
        # For an isotropic half-space, this should return the same matrix
        # as IsotropicHalfSpace
        c = p[:, 1, 0] + p[:, 1, 1]
        np.where(np.abs(c) == 0, 1, c)

        p = 2 * p / c[:, np.newaxis, np.newaxis]

        return p

    @staticmethod
    def transition_matrix_iso_halfspace(
        k_x: npt.ArrayLike, epsilon: npt.ArrayLike, inv: bool = False
    ) -> npt.NDArray:
        """Returns transition incident or exit matrix L for isotropic half-spaces.

        Args:
            k_x (npt.ArrayLike): Reduced wavenumber, Kx = kx/k0
            epsilon (npt.ArrayLike): dielectric tensor
            inv (bool, optional): If True, returns inverse transition matrix L^-1, used for the incident Matrix Li. Defaults to False.

        Returns:
            npt.NDArray: transition matrix L
        """
        n_x = sqrt(epsilon[:, 0, 0])
        sin_phi = k_x / n_x
        cos_phi = sqrt(1 - sin_phi**2)

        length = np.shape(k_x)[0]
        zeros = np.tile(0, length)
        ones = np.tile(1, length)

        if inv:
            sp_to_xy = np.array(
                [
                    [zeros, ones, -ones / cos_phi / n_x, zeros],
                    [zeros, ones, ones / cos_phi / n_x, zeros],
                    [ones / cos_phi, zeros, zeros, ones / n_x],
                    [-ones / cos_phi, zeros, zeros, ones / n_x],
                ],
                dtype=np.complex128,
            )
            return np.moveaxis(0.5 * sp_to_xy, 2, 0)

        sp_to_xy = np.array(
            [
                [zeros, zeros, cos_phi, -cos_phi],
                [ones, ones, zeros, zeros],
                [-n_x * cos_phi, n_x * cos_phi, zeros, zeros],
                [zeros, zeros, n_x, n_x],
            ],
            dtype=np.complex128,
        )
        return np.moveaxis(sp_to_xy, 2, 0)

    @staticmethod
    def get_k_z(
        material: "Material", lbda: npt.ArrayLike, k_x: npt.ArrayLike
    ) -> npt.NDArray:
        """Calculates Kz in a material

        Args:
            material (Material): Material of the half-space
            lbda (npt.ArrayLike): Wavelengths to evaluate (nm)
            k_x (npt.ArrayLike): Reduced wavenumber, Kx = kx/k0

        Returns:
            npt.NDArray: value of Kz in the material
        """
        nx = material.get_refractive_index(lbda)[:, 0, 0]
        k_z2 = nx**2 - k_x**2
        return sqrt(k_z2)

    def __init__(
        self,
        experiment: "Experiment",
        propagator: Propagator = PropagatorExpm(),
        save_experiment: bool = False,
    ) -> None:
        super().__init__(experiment, save_experiment)
        self.propagator = propagator

    def calculate(self) -> Result:
        """Calculates transition matrices for every element in the structure and resulting Jones matrices.

        Returns:
            Result: Result object with calculation results
        """
        # Kx = kx/k0 = n sin(Φ) : Reduced wavenumber.
        nx = self.structure.front_material.get_refractive_index(self.lbda)[:, 0, 0]
        k_x = nx * np.sin(np.deg2rad(self.theta_i))

        layers = reversed(self.permittivity_profile[1:-1])

        if isinstance(self.structure.back_material, IsotropicMaterial):
            m_t = self.transition_matrix_iso_halfspace(
                k_x, self.permittivity_profile[-1][1]
            )
        else:
            m_t = self.transition_matrix_halfspace(
                self.build_delta_matrix(k_x, self.permittivity_profile[-1][1])
            )

        for thickness, epsilon in layers:
            m_p = self.propagator.calculate_propagation(
                self.build_delta_matrix(k_x, epsilon), -thickness, self.lbda
            )
            m_t = m_p @ m_t

        m_lf = self.transition_matrix_iso_halfspace(
            k_x, self.permittivity_profile[0][1], inv=True
        )
        m_t = m_lf @ m_t

        # Extraction of t_it out of m_t. "2::-2" means integers {2,0}.
        t_it = m_t[:, 2::-2, 2::-2]
        # Calculate the inverse and make sure it is a matrix.
        t_ti = np.linalg.inv(t_it)

        # Extraction of t_rt out of m_t. "3::-2" means integers {3,1}.
        t_rt = m_t[:, 3::-2, 2::-2]

        # Then we have t_ri = t_rt * t_ti
        t_ri = t_rt @ t_ti

        jones_matrix_t = t_ti
        jones_matrix_r = t_ri

        # The power transmission coefficient is the ratio of the 'z' components
        # of the Poynting vector:       t = P_t_z / P_i_z
        # For isotropic media, we have: t = kb'/kf' |t_bf|^2
        # The correction coefficient is kb'/kf'
        # Note : For the moment it is only meaningful for isotropic half spaces.
        if isinstance(self.structure.back_material, IsotropicMaterial):
            k_z_f = self.get_k_z(self.structure.front_material, self.lbda, k_x)
            k_z_b = self.get_k_z(self.structure.back_material, self.lbda, k_x)
            power_correction = k_z_b.real / k_z_f.real
            return Result(
                self.experiment, jones_matrix_r, jones_matrix_t, power_correction
            )

        return Result(self.experiment, jones_matrix_r, jones_matrix_t)
