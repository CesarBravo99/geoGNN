#!/bin/bash

EXIT_CODE=0

echo "Paperspace setup - Starting Python setup"
cp -a /storage/dist-packages/. /usr/local/lib/python3.9/dist-packages/
echo "Done!"


exit $EXIT_CODE
