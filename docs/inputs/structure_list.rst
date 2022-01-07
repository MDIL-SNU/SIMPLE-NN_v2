==============
structure_list
==============

**str_list** contains the location of reference calculation data. The format is described below::

    [ structure_type_1 ]
    /location/of/calculation/data/oneshot_output_file :
    /location/of/calculation/data/MDtrajectory_output_file 100:2000:20

    [ structure_type_2 : 3.0 ]
    /location/of/calculation/data/same_folder_format{1..10}/oneshot_output_file :

You can use the format of `braceexpand`_ to set a path to reference file (like last line).
The part which is written after the path indicates the index of snapshots.
(format is 'start:end:interval'. ':' means all snapshots.)
You can group structures like above for convenience ([ structure_group_name ] above the pathes of reference file).
If ``print_structure_rmse`` is true, RMSEs for each structure type are also prited in LOG file.
In addition, you can set the weights for each structure type ([ structure_group_name : weights ], default: 1.0).

.. _braceexpand: https://pypi.org/project/braceexpand/
