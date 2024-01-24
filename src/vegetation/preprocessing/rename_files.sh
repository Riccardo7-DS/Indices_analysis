#!/bin/bash
# Create a temporary directory for storing intermediate files
input_dir="/media/BIFROST/N2/Riccardo/MSG/msg_data/NDVI/archive.eumetsat.int/umarf-gwt/onlinedownload/riccardo7/4859700/temp/"

output_dir="$input_dir/time"
variables=("NDVImean")
echo $output_dir

# Loop through each .nc file in the input directory
for variable in "${variables[@]}"; do
    for input_file in "$input_dir"/*$variable.nc; do
        #ncdump -h $input_file
        # Extract the date string from the filename using a regular expression
        date_str=$(basename "$input_file" | grep -oE '[0-9]{12}')

        #echo $date_str
        formatted_date=$(python -c "from datetime import datetime; print(datetime.strptime('$date_str', '%Y%m%d%H%M%S').strftime('%Y-%m-%d'))")
        #echo $formatted_date
        # Use cdo to add the time dimension and set the time value
        output_file="$output_dir/$(basename "$input_file" .nc)_time.nc"

        #echo $output_file
        cdo -setreftime,1900-01-01,00:00:00,1day -settaxis,$formatted_date,12:00:00,1day -setcalendar,standard "$input_file" "$output_file"

        echo "Processed: $input_file"
    done
done