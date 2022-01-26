=========
scale_rho
=========

- atom_type: ``atomic density``

----
    
The usage of **scale_rho** is as below. Users can give **scale_rho** value as atomic density(# of atoms / volume) for each atom. The unit of **scale_rho** is :math:`\mathrm{\AA ^{-3}}` 
    
.. code-block:: yaml
    
    #input.yaml
    preprocessing:
        scale_type: uniform gas
        scale_rho:
            Si: 0.01
            O : 0.02    

