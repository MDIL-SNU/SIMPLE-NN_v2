from simple_nn import Simple_nn
from simple_nn.features.symmetry_function import Symmetry_function

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()

# Main test code
symf_params_set = descriptor._parsing_symf_params()
print(sym)

"""
descriptor._get_structrue_info(snapshot, structure_tags, structure_weights)
descriptor._init_result(type_num, structure_tags, structure_weights, idx, atom_i)
descriptor._init_sf_variables(type_idx, jtem, symf_params_set, atom_num, mpi_range = None )
descriptor._check_error(errnos)
descriptor._set_result(result, x, dx, da,  type_num, jtem, symf_params_set, atom_num)
descriptor._set_EFS(result, snapshot)

descriptor.generate()
"""