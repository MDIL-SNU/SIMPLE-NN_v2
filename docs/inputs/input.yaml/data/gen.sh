list="
struct_list
refdata_format
struct_list
compress_outcar
save_directory
save_list
absolute_path
read_force
read_stress 
dx_save_sparse
"

for i in $list; do
    num=${#i}
    line="="
    for ((j=1; j<${num}; ++j)); do
        line="${line}="
    done
    cat > ${i}.rst << EOF
${line}
${i}
${line}

Introduction
============

Under construction
EOF
done
