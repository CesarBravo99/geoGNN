#!/bin/bash

EXIT_CODE=0
echo "Paperspace setup - Starting notebook setup"
PIP_DISABLE_PIP_VERSION_CHECK=1 
echo "Paperspace setup - Starting Jupyter kernel"
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True \
            --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
            --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
exit $EXIT_CODE
