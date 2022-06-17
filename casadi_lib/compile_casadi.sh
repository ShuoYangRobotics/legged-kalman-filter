search_dir=matlab_code_gen
for entry in "$search_dir"/*
do
  base_name=$(basename ${entry})

    if [ -f "${base_name%%.*}.so" ]; then
        echo "${base_name%%.*}.so exists."
    else 
        g++ -fPIC -shared -O3 "$search_dir"/"${base_name%%.*}.c" -o "${base_name%%.*}.so"
    fi
  cp "${base_name%%.*}.so" /tmp
done

