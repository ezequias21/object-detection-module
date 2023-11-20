#!/bin/bash

process_name="$OD_Module"
directory_OD_Program="$PATH_OD_Module"

init_OD_Module() {
    pid=$(pgrep -f "$process_name")

    if [ -z "$pid" ]; then
        # Move para o diretório específico
        cd "$directory_OD_Program" || exit

        # Python command
        comando_python="python3 $process_name -s"

        # Executing python command
        $comando_python
        echo "Script executing..."
    fi
}

echo "Started OD Agent"

# Check camera conectivity
if [ -n "$(ls /dev/video* 2>/dev/null)" ]; then
    echo "Camera detected."
    init_OD_Module
else
    echo "Camera not detected"
    exit 1
fi
