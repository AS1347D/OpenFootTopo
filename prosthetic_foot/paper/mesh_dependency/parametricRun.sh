
# For looping

for meshsize in $(seq 0.8 0.2 2.4); do
    # This generates the suffix for the filename
    python3 mesh.py -lc $meshsize 
    suffix=$(python3 -c "print(int(float('$meshsize')*100))")
    python3 main.py -save_log "${suffix}"
done