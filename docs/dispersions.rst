===========
Dispersions
===========

The dispersions are the central part of pyElli and the transfer-matrix method.
They describe the change of dielectric function or refractive index with the wavelength.
In pyElli the default wavelength unit is nm.
Each dispersion has two distinct sets of parameters:

    * Parameters which can be given only once (single parameters).
    * Parameters which can be given in multiple sets (repeated parameters),
      e.g. a set of oscillator parameters.

The syntax for each of the parameter sets is different.
For the single parameters they are given in the class constructor:

    .. highlight:: python
    .. code-block:: python

        Cauchy(n0=1.458, n1=3.54e-3, n2=0, k0=0, k1=0, k2=0)

Repeated parameters are added via the add() function:

    .. highlight:: python
    .. code-block:: python

        Sellmeier().add(A=1, B=1).add(A=1, B=2)

For dispersions having both, single and repeated parameters can be used together:

    .. highlight:: python
    .. code-block:: python

        TaucLorentz(Eg=2).add(A=10, E=2.5, C=0.1)

If parameters are not fully provided, they are set to their respective default values.
The available parameters and their respective default values
are given in the respective class documentation.

All classes inherit from the abstract base class `Dispersion`_.
It provides basic functionality, such as returning dataframes or arrays
containing the wavelength dependent dielectric function of the
dispersion relation at current parameter set.

Dispersions can be added with the `+` operator, or if you want to chain
more than two dispersions together you may have a look at the `DispersionSum`_ class.

PyElli also provides tabulated dispersions from the Refractiveindex.info database.
They can be accessed with the :class:`RII<elli.database.RII>` class.

Refractive Index Dispersions
============================

Constant Refractive Index
-------------------------
.. autoclass:: elli.dispersions.ConstantRefractiveIndex

Cauchy
------
.. autoclass:: elli.dispersions.Cauchy

.. plotly::

   fig = elli.Cauchy(n0=1.45, n1=36).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.data[1].visible = "legendonly"
   fig.update_layout(title="Cauchy dispersion with n0=1.45 and n1=36")

.. autoclass:: elli.dispersions.CauchyUrbach

.. plotly::

   fig = elli.CauchyUrbach(n0=1.45, B=0.005, D=0.0001).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="CauchyUrbach dispersion with n0=1.45, B=0.005, D=0.0001, Eg=2 and Eu=0.5")

.. autoclass:: elli.dispersions.CauchyCustomExponent

Dielectric Function Dispersions
===============================

Epsilon Infinity
----------------
.. autoclass:: elli.dispersions.EpsilonInf

Sellmeier
---------
.. autoclass:: elli.dispersions.Sellmeier

.. plotly::

   fig = elli.Sellmeier().add(A=1, B=1e-2).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.data[1].visible = "legendonly"
   fig.update_layout(title="Sellmeier dispersion with A=1 and B=1e-2")

.. autoclass:: elli.dispersions.SellmeierCustomExponent

Drude
-----

Energy parameters
~~~~~~~~~~~~~~~~~
.. autoclass:: elli.dispersions.DrudeEnergy

.. plotly::

   fig = elli.DrudeEnergy(A=10, gamma=0.1).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Drude dispersion with A=10 and gamma=0.1")

Resistivity parameters
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: elli.dispersions.DrudeResistivity

.. plotly::

   fig = elli.DrudeResistivity(rho_opt=1e-8, tau=1e-8).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Drude dispersion with rho_opt=tau=1e-8")

Lorentz
-------

Wavelength parameters
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: elli.dispersions.LorentzLambda

.. plotly::

   fig = elli.LorentzLambda().add(1, 500, 100).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Lorentz dispersion with A=1, lambda_r=500 and gamma=100")


Energy parameters
~~~~~~~~~~~~~~~~~
.. autoclass:: elli.dispersions.LorentzEnergy

.. plotly::

   fig = elli.LorentzEnergy().add(1, 2, 0.2).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Lorentz dispersion with A=1, E=2 and gamma=0.2")

Tauc-Lorentz
------------
.. autoclass:: elli.dispersions.TaucLorentz

.. plotly::

   fig = elli.TaucLorentz(Eg=2).add(20, 2.2, 1).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Tauc-Lorentz dispersion with Eg=2 and A=20, E=2.2 and C=1")

Cody-Lorentz
------------
.. autoclass:: elli.dispersions.CodyLorentz

.. plotly::

   fig = elli.CodyLorentz().get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Cody-Lorentz dispersion with default values")


Gaussian
--------
.. autoclass:: elli.dispersions.Gaussian

.. plotly::

   fig = elli.Gaussian().add(1, 2, 0.2).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Gaussian dispersion with A=1, E=2 and sigma=0.2")

Tanguy
------
.. autoclass:: elli.dispersions.Tanguy

.. plotly::

   fig = elli.Tanguy(Eg=2.5).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.update_layout(title="Tanguy dispersion with Eg=2.5 and default values")

Poles
-----
.. autoclass:: elli.dispersions.Poles

.. plotly::

   fig = elli.Poles(A_ir=10, A_uv=100, E_uv=8).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.data[1].visible = "legendonly"
   fig.update_layout(title="Poles dispersion with A_ir=10, A_uv=100, E_uv=8")

Polynomial
----------
.. autoclass:: elli.dispersions.Polynomial

.. plotly::

   fig = elli.Polynomial(e0=2.5).add(f=500, e=-2).get_dielectric_df().plot(backend="plotly")
   fig.update_xaxes(title="Wavelength (nm)")
   fig.update_yaxes(title="Dielectric function")
   fig.data[1].visible = "legendonly"
   fig.update_layout(title="Polynomial dispersion with e0=2.5, f_1=500, e_1=-2")

Tabulated values
================

Refractive index values
-----------------------
.. autoclass:: elli.dispersions.Table

Epsilon values
--------------
.. autoclass:: elli.dispersions.TableEpsilon
   :members:

Pseudo dielectric function
--------------------------
.. autoclass:: elli.dispersions.PseudoDielectricFunction
   :members:

Spectraray tables
-----------------
.. autoclass:: elli.dispersions.TableSpectraRay
   :members:

Abstract classes
================
These classes serve as basic interfaces for dispersions and
convenient + generalized handling.

Dispersion
----------
This is the abstract base class which is the parent class of all dispersions.
It supplies basic functionalities, such as extracting wavelength dependent
epsilon or refractive index values.

.. autoclass:: elli.dispersions.base_dispersion.Dispersion
   :members:

DispersionFactory
-----------------
This factory class returns a fully initialized class identified
by a string.
This is useful if you determine the appropriate dispersion relation
during runtime of your program.

.. autoclass:: elli.dispersions.base_dispersion.DispersionFactory
   :members:

DispersionSum
-------------
This object can be used to add an arbitrary number of dispersions.
The overloaded `+` operator of the dispersion base class creates an instance
of this object with two classes as parameters.
If you want to chain a lot of dispersions, it may be more performant to use
this class directly.

.. autoclass:: elli.dispersions.base_dispersion.DispersionSum
   :members:


InvalidParameters
-----------------
This exception is thrown whenever a dispersion got invalid parameters.

.. autoexception:: elli.dispersions.base_dispersion.InvalidParameters
