"""Decorator functions for convenient fitting"""

# Encoding: utf-8
from sys import float_info
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from lmfit import Parameters, minimize
    import plotly.graph_objects as go
    from IPython.display import display
    from ipywidgets import widgets
except ImportError as e:
    raise ImportError(
        "This module requires lmfit, plotly, ipywidgets and ipython to be installed.\n"
        "Try installing this package with the additional fitting requirement, "
        "i.e. pip install pyElli[fitting]"
    ) from e

from ..result import Result
from ..utils import calc_pseudo_diel, calc_rho
from .decorator import FitDecorator, is_in_notebook
from .params_hist import ParamsHist


class FitRho(FitDecorator):
    """
    A class to fit psi/delta or rho based ellipsometry data with two degress of freedom
    """

    def set_psi_delta(
        self, update_exp: bool = False, update_names: bool = False
    ) -> None:
        """Sets Plot to Psi/Delta values

        Args:
            update_exp (bool, optional):
                Flag to change the experimental data as well. Defaults to False.
            update_names (bool, optional):
                Flag to change the label names. Defaults to False.
        """
        data = self.model(self.exp_data.index, self.params)
        self.fig.update_layout(yaxis_title="Ψ/Δ (°)")
        self.fig.data[2].y = data.psi
        self.fig.data[3].y = data.delta

        if update_exp:
            self.fig.data[0].y = self.exp_data.loc[:, "Ψ"]
            self.fig.data[1].y = self.exp_data.loc[:, "Δ"]

        if update_names:
            self.fig.data[0].name = "Ψ"
            self.fig.data[1].name = "Δ"
            self.fig.data[2].name = "Ψ_calc"
            self.fig.data[3].name = "Δ_calc"

    def set_rho(self, update_exp: bool = False, update_names: bool = False) -> None:
        """Sets Plot to Rho values

        Args:
            update_exp (bool, optional):
                Flag to change the experimental data as well. Defaults to False.
            update_names (bool, optional):
                Flag to change the label names. Defaults to False.
        """
        data = self.model(self.exp_data.index, self.params)
        self.fig.update_layout(yaxis_title="ρ")
        self.fig.data[2].y = data.rho.real
        self.fig.data[3].y = data.rho.imag

        if update_exp:
            exp_rho = calc_rho(self.exp_data)
            self.fig.data[0].y = exp_rho.apply(lambda x: x.real).values
            self.fig.data[1].y = exp_rho.apply(lambda x: x.imag).values

        if update_names:
            self.fig.data[0].name = "ρr"
            self.fig.data[1].name = "ρi"
            self.fig.data[2].name = "ρr_calc"
            self.fig.data[3].name = "ρi_calc"

    def set_residual(
        self,
        update_exp: bool = False,  # pylint: disable=unused-argument
        update_names: bool = False,
    ) -> None:
        """Sets plots to residual values

        Args:
            update_exp (bool, optional):
                Flag to change the experimental data as well. Defaults to False.
            update_names (bool, optional):
                Flag to change the label names. Defaults to False.
        """
        data = self.model(self.exp_data.index, self.params)
        self.fig.update_layout(yaxis_title="Residual")

        exp_rho = calc_rho(self.exp_data)
        self.fig.data[0].y = self.exp_data.loc[:, "Ψ"] - data.psi
        self.fig.data[1].y = self.exp_data.loc[:, "Δ"] - data.delta
        self.fig.data[2].y = exp_rho.apply(lambda x: x.real).values - data.rho.real
        self.fig.data[3].y = exp_rho.apply(lambda x: x.imag).values - data.rho.imag

        if update_names:
            self.fig.data[0].name = "Psi Res."
            self.fig.data[1].name = "Delta Res."
            self.fig.data[2].name = "ρr Res."
            self.fig.data[3].name = "ρi Res."

    def set_pseudo_diel(
        self, update_exp: bool = False, update_names: bool = False
    ) -> None:
        """Sets Plot to Pseudo Dielectric function values

        Args:
            update_exp (bool, optional): Flag to change the experimental data as well.
                                         Defaults to False.
            update_names (bool, optional): Flag to change the label names.
                                           Defaults to False.
        """
        data = self.model(self.exp_data.index, self.params)
        peps = calc_pseudo_diel(
            pd.DataFrame(data.rho, index=self.exp_data.index).iloc[:, 0], self.angle
        )
        self.fig.update_layout(yaxis_title="ϵ")
        self.fig.data[2].y = peps.loc[:, "ϵ1"]
        self.fig.data[3].y = peps.loc[:, "ϵ2"]

        if update_exp:
            exp_peps = calc_pseudo_diel(calc_rho(self.exp_data), self.angle)
            self.fig.data[0].y = exp_peps.loc[:, "ϵ1"]
            self.fig.data[1].y = exp_peps.loc[:, "ϵ2"]

        if update_names:
            self.fig.data[0].name = "ϵ1"
            self.fig.data[1].name = "ϵ2"
            self.fig.data[2].name = "ϵ1_calc"
            self.fig.data[3].name = "ϵ2_calc"

    def update_selection(self, change=None) -> None:
        """Update plot after selection of displayed data

        Args:
            change (dict, optional): A dictionary containing the ipywidgets change event
        """
        with self.fig.batch_update():
            update = self.update_dict.get(
                change.new if change is not None else self.selector.value
            )
            if update is not None:
                update(update_exp=True, update_names=True)

    def create_widgets(self) -> None:
        """Create ipywidgets for parameter estimation"""
        self.param_widgets = {}
        self.selector = widgets.Dropdown(
            options=["Psi/Delta", "Rho", "Pseudo Diel.", "Residual"],
            value="Psi/Delta",
            description="Display: ",
            disabled=False,
        )
        self.selector.observe(self.update_selection, names="value")

        checkboxes = []
        for param in self.params.valuesdict():
            curr_checkbox = widgets.Checkbox(
                value=self.params[param].vary,
                description_tooltip=param,
                layout=widgets.Layout(width="15px"),
                indent=False,
            )
            curr_checkbox.observe(self.set_vary_param, names="value")
            curr_widget = widgets.BoundedFloatText(
                self.params[param],
                min=(
                    float_info.min
                    if self.params[param].min == -float("inf")
                    else self.params[param].min
                ),
                max=(
                    float_info.max
                    if self.params[param].max == float("inf")
                    else self.params[param].max
                ),
                description=param,
                continuous_update=False,
            )
            curr_widget.observe(self.update_params, names="value")
            checkboxes.append(curr_checkbox)
            self.param_widgets[param] = curr_widget

        fit_button = widgets.Button(description="Fit")
        fit_button.on_click(lambda _: self.fit_button_clicked())
        button_list = [self.selector, fit_button]

        if isinstance(self.params, ParamsHist):
            undo_button = widgets.Button(description="Undo")
            undo_button.on_click(self.re_undo_button_clicked)
            redo_button = widgets.Button(description="Redo")
            redo_button.on_click(self.re_undo_button_clicked)
            init_button = widgets.Button(description="Set initial")
            init_button.on_click(lambda _: self.reset_to_init_params())

            button_list.append(undo_button)
            button_list.append(redo_button)
            button_list.append(init_button)

        combo_widget = [
            widgets.HBox([v, w])
            for v, w in zip(list(self.param_widgets.values()), checkboxes)
        ]
        if is_in_notebook():
            display(
                widgets.VBox(
                    [
                        widgets.HBox(
                            combo_widget + button_list,
                            layout=widgets.Layout(
                                width="100%",
                                display="inline-flex",
                                flex_flow="row wrap",
                            ),
                        ),
                        self.fig,
                    ]
                )
            )

    def fit_function(
        self,
        params: Parameters,
        lbda: npt.NDArray,
        rhor: npt.NDArray,
        rhoi: npt.NDArray,
    ) -> npt.NDArray:
        """The fit function to minimize the fitting problem

        Args:
            params (Parameters):
                The lmfit fitting Parameters to construct the simulation
            lbda (npt.NDArray): Wavelengths in nm
            rhor (npt.NDArray): The real part of the experimental rho
            rhoi (npt.NDArray): The imaginary part of the experimental rho

        Returns:
            npt.NDArray:
                Residual between the calculation with
                current parameters and experimental data
        """
        result = self.model(lbda, params)

        sim_rho = result.rho
        resid_rhor = rhor - sim_rho.real
        resid_rhoi = rhoi - sim_rho.imag

        return np.concatenate((resid_rhor, resid_rhoi))

    def fit(self, method="leastsq"):
        """Execute lmfit with the current fitting parameters

        Args:
            method (str, optional): The fitting method to use.
                                    Any method supported by scipys curve_fit is allowed.
                                    Defaults to 'leastsq'.

        Returns:
            Result: The fitting result
        """
        res = minimize(
            self.fit_function,
            self.params,
            args=(self.lbda, self.rhor, self.rhoi),
            method=method,
        )

        self.fitted_params = res.params
        return res

    def plot(self) -> go.Figure:
        """Plot the fit results as Psi/Delta"""
        fit_result = self.model(self.exp_data.index.to_numpy(), self.fitted_params)

        return go.FigureWidget(
            pd.concat(
                [
                    self.exp_data,
                    pd.DataFrame(
                        {"Ψ_fit": fit_result.psi, "Δ_fit": fit_result.delta},
                        index=self.exp_data.index,
                    ),
                ]
            ).plot(backend="plotly")
        )

    def plot_rho(self) -> go.Figure:
        """Plot the fit results as Rho"""
        rho = calc_rho(self.exp_data)
        fit_result = self.model(rho.index.to_numpy(), self.fitted_params)

        return go.FigureWidget(
            pd.DataFrame(
                {
                    "ρr": rho.apply(lambda x: x.real),
                    "ρi": rho.apply(lambda x: x.imag),
                    "ρcr": fit_result.rho.real,
                    "ρci": fit_result.rho.imag,
                },
                index=rho.index,
            ).plot(backend="plotly")
        )

    def get_model_data(
        self, params: Parameters = None, repr: str = "psi-delta", append_exp_data=False
    ) -> pd.DataFrame:
        """Gets the data from the provided modle with the provided parameters.
        If no parameters are provided, the fitted parameters are used
        (which default to the initial parameters if no fit has been triggered).

        Args:
            params (Parameters, optional):
                The parameters to calculate the model with.
                If not provided, the fitted parameters are used.
                Defaults to None.
            repr (str, optional):
                The representation of the model and experiment data.
                Valid values are 'psi-delta' or 'rho'. Defaults to 'psi-delta'.
            append_exp_data (bool, optional):
                Appends the experimental data if set to True.
                Defaults to False.

        Raises:
            ValueError: Raised if the representation is not a valid value.

        Returns:
            pd.DataFrame: The model results
        """
        if repr not in ["psi-delta", "rho"]:
            raise ValueError(f"Unexpected representation: {repr}")

        if params is None:
            fit_result = self.model(self.exp_data.index.to_numpy(), self.fitted_params)
            desc = "fit"
        else:
            fit_result = self.model(self.exp_data.index.to_numpy(), params)
            desc = "model"

        exp_data = {"psi-delta": self.exp_data, "rho": calc_rho(self.exp_data)}

        sim_data = {
            "psi-delta": pd.DataFrame(
                {f"Ψ_{desc}": fit_result.psi, f"Δ_{desc}": fit_result.delta},
                index=self.exp_data.index,
            ),
            "rho": pd.DataFrame(
                {f"ρ_{desc}": fit_result.rho}, index=self.exp_data.index
            ),
        }

        if append_exp_data:
            return pd.concat([exp_data[repr], sim_data[repr]], axis=1)

        return sim_data[repr]

    def __init__(
        self,
        exp_data: pd.DataFrame,
        params: Parameters,
        model: Callable[[npt.NDArray, Parameters], Result],
        angle: float = 70,
        **kwargs,
    ) -> None:
        """Intialize the psi/delta fitting class

        Args:
            exp_data (pd.DataFrame):
                The dataframe containing an experimental mueller matrix.
                It should contain 2 columns with labels Ψ and Δ.
            params (Parameters): Fitting start parameters
            model (Callable[[npt.NDArray, Parameters], Result]):
                A function taking wavelengths as first
                parameter and fitting parameters as second,
                which returns a pyEllis Result object.
                This function contains the actual model which should be fitted.
            angle (float, optional): The angle of incident of the measurement.
                                     Used to calculate the Pseudo-Dielectric function.
                                     Defaults to 70.
        """
        super().__init__()
        self.model = model

        self.exp_data = exp_data
        tmp_rho = calc_rho(self.exp_data)
        self.lbda = tmp_rho.index.to_numpy()
        self.rhor = tmp_rho.values.real
        self.rhoi = tmp_rho.values.imag

        self.params = params
        self.fitted_params = params.copy()
        self.angle = angle
        self.param_widgets = {}
        self.selector = widgets.Dropdown()
        self.last_params = None
        self.initial_params = params.copy()
        self.fit_kwargs = kwargs
        self.fig = go.FigureWidget(
            pd.concat(
                [
                    exp_data,
                    pd.DataFrame(
                        {
                            "Ψ_calc": model(exp_data.index, params).psi,
                            "Δ_calc": model(exp_data.index, params).delta,
                        },
                        index=exp_data.index,
                    ),
                ]
            ).plot(backend="plotly")
        )
        self.fig.update_layout(yaxis_title="Ψ/Δ (°)", xaxis_title="Wavelength (nm)")

        self.update_dict = {
            "Psi/Delta": self.set_psi_delta,
            "Rho": self.set_rho,
            "Pseudo Diel.": self.set_pseudo_diel,
            "Residual": self.set_residual,
        }

        self.create_widgets()


def fit(
    exp_data: pd.DataFrame, params: Parameters, angle: float = 70, **kwargs
) -> Callable[[npt.NDArray, Parameters], Result]:
    """A parameters decorator for fitting psi/delta valus. Displays an ipywidget float box for
    each fitting parameter and an interactive plot to estimate parameters.

    Args:
        exp_data (pd.DataFrame): The dataframe containing an experimental mueller matrix.
                                 It should contain 2 columns with labels Ψ and Δ.
        params (Parameters): Fitting start parameters
        angle (float, optional): The angle of incident of the measurement.
                                 Used to calculate the Pseudo-Dielectric function.
                                 Defaults to 70.

    Returns:
        Callable[[npt.NDArray, Parameters], Result]:
            fitting parameters as second,
            which returns a pyEllis Result object.
            This function contains the actual model which
            should be fitted and is automatically
            provided when used as a decorator.
    """
    return lambda model: FitRho(exp_data, params, model, angle, **kwargs)
