# Segments the given foot image and saved the segmented file into GeomDir folder
# rm plots/force_displacement.txt
# touch plots/force_displacement.txt

# for force in $(seq 0 100 3200); do
#     python3 force_displacement.py -F $force
# done


# # Parametric load displacement curve for neo-Hookean model
rm plots/force_displacement_neohookean.txt
touch plots/force_displacement_neohookean.txt

for force in $(seq 0 100 3200); do
    python3 force_displacement_neohookean.py -F $force
done

## Parametric load displacement curve for Mooney-Rivlin model
# rm plots/force_displacement_mooneyrivlin.txt
# touch plots/force_displacement_mooneyrivlin.txt

# for force in $(seq 0 100 3200); do
#     python3 force_displacement_mooneyrivlin.py -F $force
# done