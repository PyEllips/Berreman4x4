# Encoding: utf-8
"""Abstract base class and utility classes for pyElli dispersion"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit import Parameter
from numpy.lib.scimath import sqrt

from .. import dispersions


class InvalidParameters(Exception):
    """Exception for invalid dispersion parameters."""


class BaseDispersion(ABC):
    """BaseDispersion (abstract class).

    Functions provided for derived classes:
    * dielectric_function(lbda) : returns dielectric constant for wavelength 'lbda'
    """

    default_lbda_range = np.linspace(200, 1000, 801)

    @property
    @abstractmethod
    def single_params_template(self) -> dict:
        """Specifies the single parameters of the model and its default values."""

    @property
    @abstractmethod
    def rep_params_template(self) -> dict:
        """Specifies the repeated parameters of the model and its default values."""

    @staticmethod
    def _guard_invalid_params(params1, params2):
        missing_params = np.array(params1)[np.where(~np.isin(params1, params2))]

        if len(missing_params) > 0:
            missing_param_strings = ", ".join(f"{p}" for p in missing_params)
            raise InvalidParameters(f"Invalid parameter(s): {missing_param_strings}")

    @staticmethod
    def _fill_params_dict(template: dict, *args, **kwargs) -> dict:
        BaseDispersion._guard_invalid_params(list(kwargs.keys()), list(template.keys()))

        if (len(kwargs) + len(args)) > len(template):
            raise InvalidParameters("Too many parameters")

        params = template.copy()
        pos_arguments = set()

        for i, val in enumerate(args):
            key = list(template.keys())[i]
            if isinstance(val, Parameter):
                val = val.value
            params[key] = val
            pos_arguments.add(key)

        for key, value in kwargs.items():
            if key in pos_arguments:
                raise InvalidParameters(
                    f"Parameter {key} already set by positional argument"
                )
            if isinstance(value, Parameter):
                value = value.value
            params[key] = value

        return params

    def __init__(self, *args, **kwargs):
        super()
        self.rep_params = []

        self.single_params = self._fill_params_dict(
            self.single_params_template, *args, **kwargs
        )

        for param in self.single_params:
            if self.single_params[param] is None:
                raise InvalidParameters(f"Please specify parameter {param}")

        self.last_lbda = None
        self.hash_single_params = None
        self.hash_rep_params = None

    def _hash_params(self, params: Union[dict, List[dict]]) -> int:
        """Creates an single_params_dict or the repeating_params_list."""
        if isinstance(params, list):
            return hash(tuple([self._hash_params(dictionary) for dictionary in params]))
        else:
            return hash(tuple([item for _, item in params.items()]))

    @abstractmethod
    def dielectric_function(self, lbda: npt.ArrayLike) -> npt.NDArray:
        """Calculates the dielectric function in a given wavelength window.

        Args:
            lbda (npt.ArrayLike): The wavelength window with unit nm.

        Returns:
            npt.NDArray: The dielectric function for each wavelength point.
        """

    def get_mat(self):
        """Returns this dispersion as an isotropic material"""
        from ..materials import IsotropicMaterial

        return IsotropicMaterial(self)

    def add(self, *args, **kwargs) -> "BaseDispersion":
        """Adds a set of parameters to the dispersion.

        Returns:
            Dispersion: The current object with the additional parameters added.
        """
        rep_param_set = self._fill_params_dict(
            self.rep_params_template, *args, **kwargs
        )
        self.rep_params.append(rep_param_set)

        return self

    def get_dielectric(self, lbda: Optional[npt.ArrayLike] = None) -> npt.NDArray:
        """Returns the dielectric constant for wavelength 'lbda' default unit (nm)
        in the convention ε1 + iε2."""
        lbda = self.default_lbda_range if lbda is None else lbda

        from .table_epsilon import TableEpsilon
        from .table_index import Table
        from .pseudo_dielectric import PseudoDielectricFunction

        if not isinstance(self, (DispersionSum, IndexDispersionSum)):
            if isinstance(self, (TableEpsilon, Table, PseudoDielectricFunction)):
                if self.last_lbda is lbda:
                    return self.cached_diel
                else:
                    self.last_lbda = lbda
                    self.cached_diel = np.asarray(
                        self.dielectric_function(lbda), dtype=np.complex128
                    )
                    return self.cached_diel
            else:
                new_single_hash = self._hash_params(self.single_params)
                new_rep_hash = self._hash_params(self.rep_params)

                if (
                    self.last_lbda is lbda
                    and self.hash_single_params == new_single_hash
                    and self.hash_rep_params == new_rep_hash
                ):
                    return self.cached_diel
                else:
                    self.last_lbda = lbda
                    self.hash_single_params = new_single_hash
                    self.hash_rep_params = new_rep_hash
                    self.cached_diel = np.asarray(
                        self.dielectric_function(lbda), dtype=np.complex128
                    )
                    return self.cached_diel

        return np.asarray(self.dielectric_function(lbda), dtype=np.complex128)

    def get_refractive_index(self, lbda: Optional[npt.ArrayLike] = None) -> npt.NDArray:
        """Returns the refractive index for wavelength 'lbda' default unit (nm)
        in the convention n + ik."""
        lbda = self.default_lbda_range if lbda is None else lbda

        if isinstance(self, IndexDispersion):
            return self.refractive_index(lbda)
        return sqrt(self.dielectric_function(lbda))

    def get_dielectric_df(
        self, lbda: Optional[npt.ArrayLike] = None, conjugate=False
    ) -> pd.DataFrame:
        """Returns the dielectric function as a pandas dataframe

        Args:
            lbda (npt.ArrayLike, optional): The wavelength range to use.
                If this parameter is not given a default spectral range
                from 200 to 1000 nm with 801 points is used.
                Defaults to None.
            conjugate (bool, optional): Per default the convention ε1 + iε2 is returned.
                If this flag is set to True, the ε1 + iε2 convention is returned.
                Defaults to False.

        Returns:
            pd.DataFrame:
                A pandas dataframe containing the wavelength as index
                and two rows containing ε1 and ε2.
        """
        lbda = self.default_lbda_range if lbda is None else lbda
        eps = self.get_dielectric(lbda)

        return pd.DataFrame(
            {"ϵ1": eps.real, "ϵ2": -eps.imag if conjugate else eps.imag},
            index=pd.Index(lbda, name="Wavelength"),
        )

    def get_refractive_index_df(
        self, lbda: Optional[npt.ArrayLike] = None, conjugate=False
    ) -> pd.DataFrame:
        """Returns the refractive index as a pandas dataframe

        Args:
            lbda (npt.ArrayLike, optional): The wavelength range to use.
                If this parameter is not given a default spectral range
                from 200 to 1000 nm with 801 points is used.
                Defaults to None.
            conjugate (bool, optional): Per default the convention n + ik is returned.
                If this flag is set to True, the n + ik convention is returned.
                Defaults to False.

        Returns:
            pd.DataFrame:
                A pandas dataframe containing the wavelength as index
                and two rows containing n and k.
        """
        lbda = self.default_lbda_range if lbda is None else lbda
        nk = self.get_refractive_index(lbda)

        return pd.DataFrame(
            {"n": nk.real, "k": -nk.imag if conjugate else nk.imag},
            index=pd.Index(lbda, name="Wavelength"),
        )

    def __repr__(self):
        def _dict_to_str(dic):
            return ", ".join(f"{item[0]} = {item[1]}" for item in dic.items())

        return (
            type(self).__name__
            + "\n"
            + "=" * len(type(self).__name__)
            + "\n"
            + _dict_to_str(self.single_params)
            + (
                "\n\nOscillators\n"
                + "===========\n"
                + "\n".join(_dict_to_str(p) for p in self.rep_params)
                if len(self.rep_params) > 0
                else ""
            )
        )


class Dispersion(BaseDispersion):
    """A dispersion based on a dielectric function formulation."""

    def _check_valid_operand(self, other: Union[int, float, "Dispersion"]):
        if not isinstance(other, (int, float, Dispersion, dispersions.TableEpsilon)):
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )

    def __radd__(self, other: Union[int, float, "Dispersion"]) -> "DispersionSum":
        """Add up the dielectric function of multiple models"""
        return self.__add__(other)

    def __add__(self, other: Union[int, float, "Dispersion"]) -> "DispersionSum":
        """Add up the dielectric function of multiple models"""
        if isinstance(other, IndexDispersion):
            raise TypeError(
                "Cannot add refractive index and dielectric function based dispersions."
            )
        self._check_valid_operand(other)

        if isinstance(other, dispersions.TableEpsilon):
            return other.__add__(self)

        if isinstance(other, DispersionSum):
            other.dispersions.append(self)
            return other

        if isinstance(other, (int, float)):
            return DispersionSum(self, dispersions.EpsilonInf(other))

        return DispersionSum(self, other)

    def as_index(self):
        """
        Returns this class as IndexDispersion.
        This method may be used to add dielectric and index based dispersions.
        Please ensure that you know what you are doing as building dielectric
        and index based dispersions is normally mathematically wrong.
        """
        index_class = deepcopy(self)
        # pylint: disable=attribute-defined-outside-init
        index_class.refractive_index = lambda lbda: sqrt(
            index_class.dielectric_function(lbda)
        )
        index_class.__class__ = IndexDispersion  # pylint: disable=invalid-name
        index_class.dielectric_function = self.dielectric_function

        return index_class


class IndexDispersion(BaseDispersion):
    """A dispersion based on a refractive index formulation."""

    @abstractmethod
    def refractive_index(self, lbda: npt.ArrayLike) -> npt.NDArray:
        """Calculates the refractive index in a given wavelength window.

        Args:
            lbda (npt.ArrayLike): The wavelength window with unit nm.

        Returns:
            npt.NDArray: The refractive index for each wavelength point.
        """

    def _check_valid_operand(self, other: Union[int, float, "IndexDispersion"]):
        if not isinstance(other, (int, float, IndexDispersion, dispersions.Table)):
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )

    def __radd__(
        self, other: Union[int, float, "IndexDispersion"]
    ) -> "IndexDispersionSum":
        """Add up the dielectric function of multiple models"""
        return self.__add__(other)

    def __add__(
        self, other: Union[int, float, "IndexDispersion"]
    ) -> "IndexDispersionSum":
        if isinstance(other, Dispersion):
            raise TypeError(
                "Cannot add refractive index and dielectric function based dispersions."
            )

        self._check_valid_operand(other)

        if isinstance(other, dispersions.Table):
            return other.__add__(self)

        if isinstance(other, IndexDispersionSum):
            other.index_dispersions.append(self)
            return other

        if isinstance(other, (int, float)):
            return IndexDispersionSum(self, dispersions.ConstantRefractiveIndex(other))

        return IndexDispersionSum(self, other)

    def dielectric_function(self, lbda: npt.ArrayLike) -> npt.NDArray:
        return self.refractive_index(lbda) ** 2

    def as_dielectric(self):
        """
        Returns this class as Dispersion.
        This method may be used to add dielectric and index based dispersions.
        Please ensure that you know what you are doing as building dielectric
        and index based dispersions is normally mathematically wrong.
        """
        diel_disp = deepcopy(self)
        diel_disp.__class__ = Dispersion  # pylint: disable=invalid-name
        diel_disp.dielectric_function = self.dielectric_function
        return diel_disp


class DispersionFactory:
    """A factory class for dispersion objects"""

    @staticmethod
    def get_dispersion(identifier: str, *args, **kwargs) -> Dispersion:
        """Creates a Dispersion object identified by its string name and initializes it
        with the given parameters.

        Args:
            identifier (str): Identifier of the Dispersion object, e.g. Cauchy.

        Returns:
            DispersionLaw: The Dispersion object initialized with the given parameters.
        """
        bad_identifier = ["Dispersion"]

        if hasattr(dispersions, identifier) and identifier not in bad_identifier:
            return getattr(dispersions, identifier)(*args, **kwargs)

        raise ValueError(f"No such dispersion: {identifier}")


class DispersionSum(Dispersion):
    """Represents a sum of two dispersions"""

    single_params_template: dict = {}
    rep_params_template: dict = {}
    dispersions: List[Dispersion]

    def __init__(self, *disps: Dispersion) -> None:
        super().__init__()

        self.dispersions = []
        for disp in disps:
            self += disp

    def __add__(self, other: Union[int, float, "Dispersion"]) -> "DispersionSum":
        self._check_valid_operand(other)

        if isinstance(other, DispersionSum):
            self.dispersions += other.dispersions
            return self

        if isinstance(other, (int, float)):
            self.dispersions.append(dispersions.EpsilonInf(eps=other))
            return self

        self.dispersions.append(other)
        return self

    def dielectric_function(self, lbda: npt.ArrayLike) -> npt.NDArray:
        dielectric_function = sum(
            disp.dielectric_function(lbda) for disp in self.dispersions
        )
        return np.array(dielectric_function)

    def __repr__(self):
        return (
            "DispersionSum\n"
            + "=" * 13
            + "\n\n"
            + "\n\n".join(map(str, self.dispersions))
        )


class IndexDispersionSum(IndexDispersion):
    """Represents the sum of two index dispersions"""

    single_params_template: dict = {}
    rep_params_template: dict = {}
    index_dispersions: List[IndexDispersion]

    def __init__(self, *disps: IndexDispersion) -> None:
        super().__init__()

        self.index_dispersions = []
        for disp in disps:
            self += disp

    def __add__(
        self, other: Union[int, float, "IndexDispersion"]
    ) -> "IndexDispersionSum":
        self._check_valid_operand(other)

        if isinstance(other, IndexDispersionSum):
            self.index_dispersions += other.index_dispersions
            return self

        if isinstance(other, (int, float)):
            self.index_dispersions.append(dispersions.ConstantRefractiveIndex(n=other))
            return self

        self.index_dispersions.append(other)
        return self

    def refractive_index(self, lbda: npt.ArrayLike) -> npt.NDArray:
        refractive_index = sum(
            disp.refractive_index(lbda) for disp in self.index_dispersions
        )
        return np.array(refractive_index)

    def __repr__(self):
        return (
            "IndexDispersionSum\n"
            + "=" * 13
            + "\n\n"
            + "\n\n".join(map(str, self.index_dispersions))
        )
