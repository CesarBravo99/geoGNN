#!/bin/bash

EXIT_CODE=0

echo "Paperspace setup - Starting Python setup"
cp -a /storage/dist-packages/. z/usr/local/lib/python3.9/dist-packages/

exit $EXIT_CODE
