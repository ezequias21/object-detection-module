#!/bin/bash

diretory=$PATH_OD_Module_Media

# In k byte size
limite_tamanho=5120
echo "Directory =$diretory"
remove_images_case_full_directory() {
    images_list=$(ls -l "$diretory" | mawk '{print $9}')
    for image in $images_list; do
        directory_size=$(du -s "$diretory" | mawk '{print $1}')
         # Remove olds images, if size directory is greater than a specific size
        if [ "$directory_size" -gt "$limite_tamanho" ]; then
            rm "$diretory/$image"
        else
            break
        fi
    done
}

echo "Started DS Agent"
remove_images_case_full_directory