# Encoding: utf-8
from abc import ABC, abstractmethod
from copy import deepcopy

from .result import Result


class Solver(ABC):
    """
    Solver base class to evaluate Experiment objects.
    Here the experiment and structure get unpacked
    and the simulation results get returned.

    The actual simulation is handled by subclasses.
    Therefore, this class should never be called directly.
    """

    experiment = None
    structure = None
    lbda = None
    theta_i = None
    jones_vector = None
    permittivity_profile = None

    @abstractmethod
    def calculate(self) -> Result:
        pass

    def __init__(self, experiment: "Experiment", save_experiment: bool = False) -> None:
        if save_experiment:
            self.experiment = deepcopy(experiment)
        else:
            self.experiment = experiment
        self.structure = self.experiment.structure
        self.lbda = self.experiment.lbda
        self.theta_i = self.experiment.theta_i
        self.jones_vector = self.experiment.jones_vector
        self.permittivity_profile = self.structure.get_permittivity_profile(self.lbda)
