#!/bin/bash

SQSH_FILE=$1
CONTAINER_NAME=$2

# Check if the container name already exists before creating a new one
if enroot list | grep -Fxq ${CONTAINER_NAME} ; then
    echo "Container ${CONTAINER_NAME} already exists."
else
    echo "Creating enroot container ${CONTAINER_NAME} from ${SQSH_FILE}"
    enroot create --name ${CONTAINER_NAME} ${SQSH_FILE}
fi

