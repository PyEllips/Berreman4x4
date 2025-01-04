"""Benchmark for using the formula dispersion"""

import elli
import numpy as np
from elli.fitting import ParamsHist, fit
from fixtures import datadir
from pytest import fixture


def test_fitting_structure_creation(benchmark, datadir):
    ANGLE = 70
    rii_db = elli.db.RII()
    Si = rii_db.get_mat("Si", "Aspnes")

    psi_delta = (
        elli.read_nexus_psi_delta(datadir / "SiO2onSi.ellips.nxs")
        .loc[ANGLE]
        .loc[210:800]
    )

    params = ParamsHist()
    params.add("SiO2_n0", value=1.6, min=-100, max=100, vary=True)
    params.add("SiO2_n1", value=36, min=-40000, max=40000, vary=False)
    params.add("SiO2_n2", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_k0", value=0, min=-100, max=100, vary=False)
    params.add("SiO2_k1", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_k2", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_d", value=20, min=0, max=40000, vary=True)

    @fit(psi_delta, params)
    def model(lbda, params):
        SiO2 = elli.Cauchy(
            params["SiO2_n0"],
            params["SiO2_n1"],
            params["SiO2_n2"],
            params["SiO2_k0"],
            params["SiO2_k1"],
            params["SiO2_k2"],
        ).get_mat()

        return elli.Structure(
            elli.AIR,
            [elli.Layer(SiO2, params["SiO2_d"])],
            Si,
        ).evaluate(lbda, ANGLE, solver=elli.Solver2x2)

    result = benchmark.pedantic(
        model.fit,
        args=(),
        iterations=1,
        rounds=10,
    )
    assert result.chisqr < 0.02


def test_fitting_structure_updates(benchmark, datadir):
    ANGLE = 70
    rii_db = elli.db.RII()
    Si = rii_db.get_mat("Si", "Aspnes")

    psi_delta = (
        elli.read_nexus_psi_delta(datadir / "SiO2onSi.ellips.nxs")
        .loc[ANGLE]
        .loc[210:800]
    )

    params = ParamsHist()
    params.add("SiO2_n0", value=1.6, min=-100, max=100, vary=True)
    params.add("SiO2_n1", value=36, min=-40000, max=40000, vary=False)
    params.add("SiO2_n2", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_k0", value=0, min=-100, max=100, vary=False)
    params.add("SiO2_k1", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_k2", value=0, min=-40000, max=40000, vary=False)
    params.add("SiO2_d", value=20, min=0, max=40000, vary=True)

    SiO2 = elli.Cauchy(
        params["SiO2_n0"],
        params["SiO2_n1"],
        params["SiO2_n2"],
        params["SiO2_k0"],
        params["SiO2_k1"],
        params["SiO2_k2"],
    )

    SiO2_mat = SiO2.get_mat()

    layer = elli.Layer(SiO2_mat, params["SiO2_d"])

    structure = elli.Structure(
        elli.AIR,
        [layer],
        Si,
    )

    @fit(psi_delta, params)
    def model(lbda, params):
        SiO2.single_params["n0"] = params["SiO2_n0"].value
        SiO2.single_params["n1"] = params["SiO2_n1"].value
        SiO2.single_params["n2"] = params["SiO2_n2"].value
        SiO2.single_params["k0"] = params["SiO2_k0"].value
        SiO2.single_params["k1"] = params["SiO2_k1"].value
        SiO2.single_params["k2"] = params["SiO2_k2"].value

        layer.set_thickness(params["SiO2_d"])

        return structure.evaluate(lbda, ANGLE, solver=elli.Solver2x2)

    result = benchmark.pedantic(
        model.fit,
        args=(),
        iterations=1,
        rounds=10,
    )
    assert result.chisqr < 0.02
