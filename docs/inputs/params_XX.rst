=========
params_XX
=========

**params_XX** contains the coefficients for symmetry functions (SFs). XX is the element name  
in the target system. 'param_XX' is written in the following style:

.. code-block:: bash

    2 1 0 6.0 0.003214 0.0 0.0
    2 1 0 6.0 0.035711 0.0 0.0
    4 1 1 6.0 0.000357 1.0 -1.0
    4 1 1 6.0 0.028569 1.0 -1.0
    4 1 1 6.0 0.089277 1.0 -1.0

Each line means:

.. code-block:: bash

    [Type of SF (1)] [Atom-type index (2)] [Cutoff radius (1)] [Coefficients for SF (3)]

The number inside (·) is the dimension of parameters.

[Type of SF] Currently, G2 (2), G4 (4), and G5 (5) are supported.

[Atom-type index] Type indices of neighbor atoms which starts from 1.
The order of type index follows that of the ``params`` tag written in ``input.yaml``) 
The radial part (G2) requires only one neighbor type so the second parameter is neglected. 
For the angular parts (G4 and G5), two neighboring types are needed. 
The order of the two parameters does not affect the results.

[Cutoff radius] The cutoff radius for cutoff function.

[Coefficients for SF] The parameters defining each symmetry function.
For G2, the first two values indicate :math:`\eta` and :math:`\mathrm{R_s}` and the third one is neglected.
For G4 and G5, :math:`\eta`, :math:`\zeta`, and :math:`\lambda` are listed in this order.
