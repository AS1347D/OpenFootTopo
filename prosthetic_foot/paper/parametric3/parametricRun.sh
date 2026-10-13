# Parametric run for full factorial sweep
# python3 main.py -volfrac 0.75 -condition P3 -save_log result_P3_75

# For looping

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P3 -save_log "P3_${suffix}"
done

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P4 -save_log "P4_${suffix}"
done

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P5 -save_log "P5_${suffix}"
done

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P6 -save_log "P6_${suffix}"
done

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P7 -save_log "P7_${suffix}"
done

for v in $(seq 0.15 0.05 0.8); do
    # This generates the suffix for the filename
    suffix=$(python3 -c "print(int(float('$v')*100))")
    python3 main.py -vol_frac $v -condition P8 -save_log "P8_${suffix}"
done