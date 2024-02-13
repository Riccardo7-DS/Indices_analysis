#!/bin/bash

# Define the translation and warping commands
translate_command="gdal_translate -a_srs \"+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +no_defs\" -a_ullr -5568000 5568000 5568000 -5568000 HDF5:\"input_filename\"://var temp.tif"
warp_command0="gdalwarp -t_srs EPSG:4326 -wo SOURCE_EXTRA=100 temp.tif output_filename -overwrite"

# Define the variables
variables=( "NDVImax") #"NDVImean" "NDVImin" "NDVIaccum"

# Set the path to the main directory
main_path="/media/BIFROST/N2/Riccardo/MSG/msg_data/NDVI/archive.eumetsat.int/umarf-gwt/onlinedownload/riccardo7/4859700"

# Create a temporary directory for storing intermediate files
temp_dir="/media/BIFROST/N2/Riccardo/MSG/msg_data/NDVI/archive.eumetsat.int/umarf-gwt/onlinedownload/riccardo7/4859700/temp/max"

mkdir -p $temp_dir

# Iterate over the subfolders
for i in {1..8}; do
    subfolder_path="${main_path}/4859700_${i}_of_8"

    # # Change directory to the subfolder
    cd "${subfolder_path}" || exit 1

    # Iterate over files ending with ".hf"
    for input_filename in "$subfolder_path"/*.h5; do
        # echo "----------Warped command: ${warp_command0}"
        newname=$(basename "$input_filename" | cut -f1 -d'.')
        echo "----------Filename---------: ${newname}"+
        

        # Check if there are matching files
        if [ -n "$input_filename" ]; then

            for variable in "${variables[@]}"; do
            # echo "----------Warped command 2: ${warp_command0}"
            echo "----------Converting variable ${variable}"
                # Replace "input_filename" and "output_filename" in commands
                translated_command="${translate_command//input_filename/${input_filename}}"
                translated_command="${translated_command//var/${variable}}"

                warped_output="${temp_dir}/${newname}_${variable}.nc"
                warp_command="${warp_command0//output_filename/${warped_output}}"
                # echo "----------Warped command 2: ${warp_command}"
                # Execute the translation command
                eval "${translated_command}"
                # Execute the warping command
                eval "${warp_command}"
            done
        fi        
    done
done

# Merge the NetCDF files using cdo
for variable in "${variables[@]}"; do
    cdo -O mergetime "${temp_dir}/"*"${variable}.nc" "${main_path}/merged_${variable}.nc"

done

# # Clean up temporary files
# rm -rf "${temp_dir}"