=========
params_XX
=========

**params_XX** contains the coefficients for symmetry functions. XX is an atom type which 
included in the target system. The detailed format of 'param_XX' is described in below

.. code-block:: bash

    2 1 0 6.0 0.003214 0.0 0.0
    2 1 0 6.0 0.035711 0.0 0.0
    4 1 1 6.0 0.000357 1.0 -1.0
    4 1 1 6.0 0.028569 1.0 -1.0
    4 1 1 6.0 0.089277 1.0 -1.0

Each parameter indicates (SF means symmetry function)

.. code-block:: bash

    [type of SF(1)] [atom type index(2)] [cutoff distance(1)] [coefficients for SF(3)]

The number inside (·) indicates the number of parameters.

First column indicates the type of symmetry function. Currently, G2 (2), G4 (4), and G5 (5) are available.

Second and third column indicates the type index of neighbor atoms which starts from 1.
(The order of type index need to be the same as the order of the ``params`` tag indicated in ``input.yaml``) 
For radial symmetry function, only one neighbor atom needed to calculate the symmetry function value, 
thus third parameter is set to zero. For angular symmetry function, two neighbor atoms are needed. 
The order of second and third column do not affect the calculation result.

The fourth column means the cutoff radius for cutoff function.

The remaining columns are the parameters applied to each symmetry function.
For radial symmetry function, the fifth and sixth column indicates :math:`\eta` and :math:`\mathrm{R_s}`.
The value in last column is dummy value.
For angular symmetry function, :math:`\eta`, :math:`\zeta`, and :math:`\lambda` are listed in order.
