# Encoding: utf-8
"""After a calculation is completed, the Solver returns an object of the Result class.

It is used to store the evaluated experiment and all resulting optical properties like
Jones matrices and ellipsometric parameters (psi, delta, rho, mueller matrices).
A list of all properties is given below.

All properties will return an array in the length of the provided wavelength array
of the requested property.

These can be accessed by different methods:

* With dot notation: ``result.property``
* With the get method: ``result.get('property')``

For Matrix properties, a specific value can be requested using a ``property_ij`` notation.
As i and j the respective polarization identifiers or 
numerical indices can be used (r/s or R/L or 0...3).

To make handling multiple experiments easier, they can be grouped into a list and provided
to a ResultList object. It provides the same methods for data output as the single Result.
The Output is returned as array over the list of results.
"""
from typing import List

import numpy as np
import numpy.typing as npt
from numpy.lib.scimath import sqrt


def _polar_index(index: str) -> int:
    """Return polarization index for character 'index'.

    Args:
        index (str): Polarization index, valid are: 'p', 's', 'R', 'L' and '0' till '3'.
    Returns:
        int: 'p', 'L' -> 0
                's', 'R' -> 1
    """
    if index in ["0", "1", "2", "3"]:
        return int(index)
    if index in ["p", "L"]:
        return 0
    if index in ["s", "R"]:
        return 1

    return ValueError("Wrong index given for variable.")


class Result:
    """Record of a simulation result."""

    @property
    def rho(self) -> npt.NDArray:
        r"""Returns the ellipsometric parameter :math:`\rho` in reflection direction.

        It is calculated by dot product of the \rho matrix
        :math:`M_\rho` and the Jones vector :math:`\vec{E}` of the incident light beam.
        It then takes the :math:`\rho_\text{pp}` element and returns it.

        .. math::
            M_{\text{$\rho$, exp}} = M_\rho \cdot \vec{E}
        """
        rho = np.dot(self.rho_matrix, self.experiment.jones_vector)
        rho = rho[:, 0] / rho[:, 1]
        return rho

    @property
    def rho_t(self) -> npt.NDArray:
        r"""Returns the ellipsometric parameter :math:`\rho_\text{t}` in transmission direction."""
        rho_t = np.dot(self.rho_matrix_t, self.experiment.jones_vector)
        rho_t = rho_t[:, 0] / rho_t[:, 1]
        return rho_t

    @property
    def psi(self) -> npt.NDArray:
        r"""Returns the ellipsometric angle :math:`\psi` in reflection direction.

        It results from:

        .. math::
            \rho = \tan \psi \exp(-i \Delta)
        """
        return np.rad2deg(np.arctan(np.abs(self.rho)))

    @property
    def psi_t(self) -> npt.NDArray:
        r"""Returns the ellipsometric angle :math:`\psi_\text{t}` in transmission direction.

        It results from:

        .. math::
            \rho_\text{t} = \tan \psi_\text{t} \exp(-i \Delta_\text{t})
        """
        return np.rad2deg(np.arctan(np.abs(self.rho_t)))

    @property
    def delta(self) -> npt.NDArray:
        r"""Returns the ellipsometric angle :math:`\Delta` in reflection direction.

        It results from:

        .. math::
            \rho = \tan \psi \exp(-i \Delta)
        """
        return -np.angle(self.rho, deg=True)

    @property
    def delta_t(self) -> npt.NDArray:
        r"""Returns the ellipsometric angle :math:`\Delta_\text{t}` in transmission direction.

        It results from:

        .. math::
            \rho_\text{t} = \tan \psi_\text{t} \exp(-i \Delta_\text{t})
        """
        return -np.angle(self.rho_t, deg=True)

    @property
    def rho_matrix(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\rho` in reflection direction.

        .. math::
            M_\rho = \begin{bmatrix}
            \rho_\text{pp} & \rho_\text{ps} \\ \rho_\text{sp} & 1
            \end{bmatrix}
            = r_\text{ss} \begin{bmatrix}
            r_\text{pp}/r_\text{ss} & r_{ps}/r_\text{ss} \\ r_\text{sp}/r_\text{ss} & 1
            \end{bmatrix}
        """
        r_ss = self.jones_matrix_r[..., 1, 1]
        return self.jones_matrix_r / r_ss[:, None, None]

    @property
    def rho_matrix_t(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\rho_t` in reflection direction.

        .. math::
            M_{\rho \text{,t}} = \begin{bmatrix}
            \rho_\text{t,pp} & \rho_\text{t,ps} \\ \rho_\text{t,sp} & 1
            \end{bmatrix}
            = t_\text{ss} \begin{bmatrix}
            t_\text{pp}/t_\text{ss} & t_\text{ps}/t_\text{ss} \\ t_\text{sp}/t_\text{ss} & 1
            \end{bmatrix}
        """
        t_ss = self.jones_matrix_t[..., 1, 1]
        return self.jones_matrix_t / t_ss[:, None, None]

    @property
    def psi_matrix(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\psi` in reflection direction.

        .. math::
            M_\psi = \begin{bmatrix}
            \psi_\text{pp} & \psi_\text{ps} \\ \psi_\text{sp} & 45°
            \end{bmatrix}
        """
        return np.rad2deg(np.arctan(np.abs(self.rho_matrix)))

    @property
    def psi_matrix_t(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\psi_\text{t}` in transmission direction.

        .. math::
            M_{\psi \text{,t}} = \begin{bmatrix}
            \psi_\text{t,pp} & \psi_\text{t,ps} \\ \psi_\text{t,sp} & 45°
            \end{bmatrix}
        """
        return np.rad2deg(np.arctan(np.abs(self.rho_matrix_t)))

    @property
    def delta_matrix(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\Delta` in reflection direction.

        .. math::
            M_\Delta = \begin{bmatrix}
            \Delta_\text{pp} & \Delta_\text{ps} \\ \Delta_\text{sp} & 0°
            \end{bmatrix}
        """
        return -np.angle(self.rho_matrix, deg=True)

    @property
    def delta_matrix_t(self) -> npt.NDArray:
        r"""Returns the matrix of the ellipsometric parameter
        :math:`\Delta_\text{t}` in transmission direction.

        .. math::
            M_{\Delta \text{,t}} = \begin{bmatrix}
            \Delta_\text{t,pp} & \Delta_\text{t,ps} \\ \Delta_\text{t,sp} & 0°
            \end{bmatrix}
        """
        return -np.angle(self.rho_matrix_t, deg=True)

    @property
    def mueller_matrix(self) -> npt.NDArray:
        """Returns the Mueller matrix for reflection, calculated from the rho matrix."""
        a = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])

        # Kronecker product of S and S*
        s_kron_s_star = np.einsum(
            "aij,akl->aikjl", np.conjugate(self.rho_matrix), self.rho_matrix
        ).reshape([self.rho_matrix.shape[0], 4, 4])

        mueller_matrix = np.real(a @ s_kron_s_star @ np.linalg.inv(a))
        mm11 = mueller_matrix[:, 0, 0]

        return mueller_matrix / mm11[:, None, None]

    @property
    def jones_matrix_r(self) -> npt.NDArray:
        r"""Returns the Jones matrix with the amplitude reflection coefficients.

        .. math::
            M_\text{r} = \begin{bmatrix}
            r_\text{pp} & r_\text{ps} \\ r_\text{sp} & r_\text{ss}
            \end{bmatrix}
        """
        return self._jones_matrix_r

    @property
    def jones_matrix_t(self) -> npt.NDArray:
        r"""Returns the Jones matrix with the amplitude transmission coefficients.

        .. math::
            M_\text{t} = \begin{bmatrix}
            t_\text{pp} & t_\text{ps} \\ t_\text{sp} & t_\text{ss}
            \end{bmatrix}
        """
        return self._jones_matrix_t

    @property
    def jones_matrix_rc(self) -> npt.NDArray:
        r"""Returns the Jones matrix with the amplitude reflection coefficients
        for circular polarization.

        .. math::
            M_\text{rc} = \begin{bmatrix}
            r_\text{LL} & r_\text{LR} \\ r_\text{RL} & r_\text{RR}
            \end{bmatrix}
        """
        c = 1 / sqrt(2) * np.array([[1, 1], [1j, -1j]])
        d = 1 / sqrt(2) * np.array([[-1, -1], [-1j, 1j]])
        return np.einsum(
            "ij,...jk,kl->...il", np.linalg.inv(d), self._jones_matrix_r, c
        )

    @property
    def jones_matrix_tc(self) -> npt.NDArray:
        r"""Returns the Jones matrix with the amplitude transmission coefficients
        for circular polarization.

        .. math::
            M_\text{tc} = \begin{bmatrix}
            t_\text{LL} & t_\text{LR} \\ t_\text{RL} & t_\text{RR}
            \end{bmatrix}
        """
        c = 1 / sqrt(2) * np.array([[1, 1], [1j, -1j]])
        return np.einsum(
            "ij,...jk,kl->...il", np.linalg.inv(c), self._jones_matrix_t, c
        )

    @property
    def R(self) -> npt.NDArray:
        r"""Returns the reflectance matrix separated for s and p polarization.

        .. math::
            M_R = \begin{bmatrix} R_{pp} & R_{ps} \\ R_{sp} & R_{ss} \end{bmatrix}
        """
        return np.abs(self._jones_matrix_r) ** 2

    @property
    def T(self) -> npt.NDArray:
        r"""Returns the transmittance matrix separated for s and p polarization.

        .. math::
            M_T = \begin{bmatrix} T_{pp} & T_{ps} \\ T_{sp} & T_{ss} \end{bmatrix}
        """
        return np.abs(self._jones_matrix_t) ** 2 * self._power_correction[:, None, None]

    @property
    def Rc(self) -> npt.NDArray:
        r"""Returns the reflectance matrix with the for circular polarizations.

        .. math::
            M_{Rc} = \begin{bmatrix} R_{LL} & R_{LR} \\ R_{RL} & R_{RR} \end{bmatrix}
        """
        return np.abs(self.jones_matrix_rc) ** 2

    @property
    def Tc(self) -> npt.NDArray:
        r"""Returns the transmittance matrix with the for circular polarizations.

        .. math::
            M_{Tc} = \begin{bmatrix} T_{LL} & T_{LR} \\ T_{RL} & T_{RR} \end{bmatrix}
        """
        return np.abs(self.jones_matrix_tc) ** 2 * self._power_correction[:, None, None]

    def __init__(
        self,
        experiment: "Experiment",
        jones_matrix_r: npt.NDArray,
        jones_matrix_t: npt.NDArray,
        power_correction: npt.NDArray = None,
    ) -> None:
        """Creates result object, to store simulation data. Gets called by solvers.

        Args:
            experiment (Experiment):
                Evaluated experiment, with structure and experimental parameters.
            jones_matrix_r (npt.NDArray): Jones matrix for the reflection direction.
            jones_matrix_t (npt.NDArray): Jones matrix for the transmission direction.
            power_correction (npt.NDArray):
                Correction factors, to get the power transmission values.
        """
        self.experiment = experiment
        self._jones_matrix_r = jones_matrix_r
        self._jones_matrix_t = jones_matrix_t
        if power_correction is None:
            self._power_correction = np.ones_like(self.experiment.lbda)
        else:
            self._power_correction = power_correction

    def get(self, name: str) -> npt.NDArray:
        """Return the data for the requested variable 'name'.

        Args:
            name (str): Variable name to return.
                Examples for 'name':

                * 'r_sp' : Amplitude reflection coefficient from 's' to 'p' polarization.
                * 'r_LR' : Reflection from circular right to circular left polarization.
                * 'T_pp' : Power transmission coefficient from 'p' to 'p' polarization.
                * 'Ψ_ps', 'Δ_pp' : Ellipsometry parameters.
                * 'psi', 'delta', 'rho': Reduced ellipsometry parameters,
                  the whole matrices are returned by 'psi_matrix'.

        Returns:
            npt.NDArray: Array of data.
        """
        return self[name]

    def __getattr__(self, name: str) -> npt.NDArray:
        """Return the data for the requested variable 'name'.

        Args:
            name (str): Variable name to return.
                Examples for 'name'...
                'r_sp' : Amplitude reflection coefficient from 's' to 'p' polarization.
                'r_LR' : Reflection from circular right to circular left polarization.
                'T_pp' : Power transmission coefficient from 'p' to 'p' polarization.
                'Ψ_ps', 'Δ_pp' : Ellipsometry parameters.
                'psi', 'delta', 'rho':
                    Reduced ellipsometry parameters,
                    the whole matrices are returned by 'psi_matrix'.

        Returns:
            npt.NDArray: Array of data.
        """
        names = name.rsplit("_", 1)

        if names[0] == "Ψ":
            names[0] = "psi"
        elif names[0] == "Δ":
            names[0] = "delta"
        elif names[0] == "ρ":
            names[0] = "rho"

        if not (
            names[0] in ["psi", "delta", "rho", "r", "t", "rc", "tc"]
            or names[0] in self.__dir__()
        ):
            raise AttributeError(f"'Result' object has no attribute '{name}'")

        if len(names) > 1:
            (i, j) = map(_polar_index, names[1])

        if names[0] in ["psi", "delta", "rho"]:
            if len(names) == 1:
                return self.__getattribute__(names[0])
            return self.__getattribute__(names[0] + "_matrix")[:, i, j]

        if names[0] in ["r", "rc", "t", "tc"]:
            if len(names) == 1:
                return self.__getattribute__("jones_matrix_" + names[0])
            return self.__getattribute__("jones_matrix_" + names[0])[:, i, j]

        return self.__getattribute__(names[0])[:, i, j]


class ResultList:
    """Class to make a row of Results easier to handle."""

    def __init__(self, results: List[Result] = None) -> None:
        """Creates an ResultList object.

        Args:
            results (List[Result], optional): List of results to store. Defaults to None.
        """
        if results is None:
            self.results = []
        else:
            self.results = results

    def append(self, result: Result) -> None:
        """Append a single Result to the ResultList.

        Args:
            result (Result): Additional Result to store.
        """
        self.results.append(result)

    def __len__(self) -> int:
        """Returns length of ResultList.

        Returns:
            int: Number of Results in ResultList.
        """
        return len(self.results)

    def __getattr__(self, name: str) -> npt.NDArray:
        """Returns the data for the requested variable 'name' of all results.

        Args:
            name (str): Variable name to return.
                Examples for 'name'...
                'r_sp' : Amplitude reflection coefficient from 's' to 'p' polarization.
                'r_LR' : Reflection from circular right to circular left polarization.
                'T_pp' : Power transmission coefficient from 'p' to 'p' polarization.
                'Ψ_ps', 'Δ_pp' : Ellipsometry parameters.
                'psi', 'delta', 'rho':
                    Reduced ellipsometry parameters,
                    the whole matrices are returned by 'psi_matrix'.

        Returns:
            npt.NDArray: Array of data.
        """

        return np.squeeze(np.array([getattr(result, name) for result in self.results]))
