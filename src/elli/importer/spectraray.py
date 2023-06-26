"""A helper class to load data from SpectraRay ASCII Files.
It only supplies a rudimentary loading of standard psi/delta values
and misses some other features.
"""
import pandas as pd

from ..utils import calc_rho


def read_spectraray_psi_delta(
    fname: str, sep: str = " ", decimal: str = "."
) -> pd.DataFrame:
    r"""Read a psi/delta spectraray ascii file.
    Only reads the first entry and does not support reading multiple angles.
    For multiple angles you have to save the data in multiple files.

    Args:
        fname (str): Filename of the measurement ascii file.
        sep (str, optional): Data separator in the datafile. Defaults to " ".
        decimal (str, optional): Decimal separator in the datafile. Defaults to ".".

    Returns:
        pd.DataFrame: DataFrame containing the psi/delta data in
        the format to be further processes inside pyElli.
    """

    # read data and drop empty column
    psi_delta_df = pd.read_csv(
        fname,
        index_col=0,
        header=None,
        sep=sep,
        decimal=decimal,
        skiprows=1,
    )
    psi_delta_df.dropna(axis="columns", how="all", inplace=True)

    # index data correctly
    psi_delta_df.index.name = "Wavelength"

    with open(fname) as f:
        header = f.readlines()[0]

    aois = list(map(float, header.split(sep)[2::2]))
    index = pd.MultiIndex.from_product(
        [aois, ["Ψ", "Δ"]], names=["Angle of Incidence", ""]
    )
    psi_delta_df.columns = index

    # reorder dataframe
    psi_delta_df = psi_delta_df.stack(0)
    psi_delta_df = psi_delta_df.reorder_levels(["Angle of Incidence", "Wavelength"])
    psi_delta_df.sort_index(axis=0, inplace=True)
    psi_delta_df.sort_index(axis=1, ascending=False, inplace=True)

    # convert delta range
    psi_delta_df.loc[:, "Δ"] = psi_delta_df.loc[:, "Δ"].where(
        psi_delta_df.loc[:, "Δ"] <= 180, psi_delta_df.loc[:, "Δ"] - 360
    )

    return psi_delta_df


def read_spectraray_mmatrix(
    fname: str, sep: str = r" ", decimal: str = "."
) -> pd.DataFrame:
    r"""Read a mueller matrix spectraray ascii file.
    Only reads the first entry and does not support reading multiple angles.
    For multiple angles you have to save the data in multiple files.

    Args:
        fname (str): Filename of the measurement ascii file.
        sep (str, optional): Data separator in the datafile. Defaults to " ".
        decimal (str, optional): Decimal separator in the datafile. Defaults to ".".

    Returns:
        pd.DataFrame: DataFrame containing the psi/delta data in
        the format to be further processes inside pyElli.
    """
    mueller_matrix = pd.read_csv(fname, sep=sep, decimal=decimal, index_col=0).iloc[
        :, -17:-1
    ]
    mueller_matrix.index.name = "Wavelength"
    mueller_matrix.columns = [
        "M11",
        "M12",
        "M13",
        "M14",
        "M21",
        "M22",
        "M23",
        "M24",
        "M31",
        "M32",
        "M33",
        "M34",
        "M41",
        "M42",
        "M43",
        "M44",
    ]

    return mueller_matrix


def read_spectraray_rho(
    fname: str, sep: str = r" ", decimal: str = "."
) -> pd.DataFrame:
    r"""Read a psi/delta spectraray ascii file and converts it to rho values.
    Only reads the first entry and does not support reading multiple angles.
    For multiple angles you have to save the data in multiple files.

    Args:
        fname (str): Filename of the measurement ascii file.
        sep (str, optional): Data separator in the datafile. Defaults to " ".
        decimal (str, optional): Decimal separator in the datafile. Defaults to ".".

    Returns:
        pd.DataFrame: DataFrame containing the rho data in
        the format to be further processes inside pyElli.
    """
    psi_delta = read_spectraray_psi_delta(fname, sep, decimal)
    return calc_rho(psi_delta)
