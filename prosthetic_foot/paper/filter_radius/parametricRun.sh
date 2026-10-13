# Parametric run for full factorial sweep
# python3 main.py -volfrac 0.75 -condition P3 -save_log result_P3_75

# For looping

# for R in $(seq 5 2.5 15); do
#     # This generates the suffix for the filename
#     suffix=$(python3 -c "print(int(float('$R')))")
#     python3 main.py -filter_radius $R -condition P5 -save_log "P5_${suffix}" -results_dir "iterations_${suffix}"
# done

R=2.5
suffix=$(python3 -c "print(int(float('$R')))")
python3 main.py -filter_radius $R -condition P5 -save_log "P5_${suffix}" -results_dir "iterations_${suffix}"