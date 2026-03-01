#! /bin/bash

# Clear logs in data dirs
PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

for dir in $PROJ_DIR/data/*; do
    if [ -d "$dir" ]; then
        rm -rf "$dir/logs/*"
    fi
done

echo "Logs cleared"