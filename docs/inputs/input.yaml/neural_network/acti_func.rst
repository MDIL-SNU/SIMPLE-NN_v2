=========
acti_func
=========

- ``sigmoid`` (default) / ``tanh``, ``relu``, ``selu``, ``swish`` 

----

SIMPLE-NN supports several activation functions, such as ``sigmoid`` function which is the default setting, hyperbolic tangent(``tanh``) function, rectified linear unit(``relu``) function, scaled exponential linear unit(``selu``) function and ``swish`` function. The usage of **acti_func** is as below.

.. code-block:: yaml

    # input.yaml
    neural_network:
       acti_func: sigmoid 
