#!/bin/bash

# List of required Python packages
REQUIRED_PKGS=(
    torch
    pyfaidx
    biopython
    scikit-learn
    pandas
    numpy
)

echo "Checking required Python packages..."

for pkg in "${REQUIRED_PKGS[@]}"
do
    echo -n "Checking $pkg... "
    python -c "import $pkg" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "not found. Installing..."
        pip install "$pkg"
    else
        echo "ok"
    fi
done

echo "All required packages are installed."
