#!/bin/bash
#========================#
#  AIkido by @JoelGMSec  #
#      darkbyte.net      #
#========================#

if docker -v &> /dev/null; then
    if ! (( $(ps -ef | grep -v grep | grep dockerd | wc -l) > 0 )); then
        sudo service docker start > /dev/null 2>&1
        sleep 2
    fi
fi

if ! sudo docker image inspect joelgmsec/aikido > /dev/null 2>&1; then
     sudo docker build -t joelgmsec/aikido .
fi

sudo docker run --rm --net host -it -v "$(pwd)":/AIkido joelgmsec/aikido "$@"
