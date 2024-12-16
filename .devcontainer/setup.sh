#!/usr/bin/env sh
python3 -m pip install --upgrade pip pipx --user
python3 -m pipx ensurepath
BP="$HOME/.bash_profile"
ZP="$HOME/.zprofile"
FOUND=""

# Check if $BP is non-empty and valid
if [ -n "${BP}" ] && [ -f "${BP}" ]; then
    echo "Source bash profile"
    FOUND="${BP}"

# Check if $ZP is non-empty and valid
elif [ -n "${ZP}" ] && [ -f "${ZP}" ]; then
    echo "Source zprofile"
    FOUND="${ZP}"

# If neither file exists
else
    echo "No profile found, exiting..."
    exit 1
fi

# By now, found can't be empty.
. "${FOUND}"

# Get uv
pipx install uv


# Remove old venv if it's there
if [ -d "/.venv" ]; then
    rm -rf .venv
fi

uv venv --python 3.10 --verbose
# `source` not available from scripts
. .venv/bin/activate
# for Apple Silicon, fallsback to PyPI 
uv sync --extra cpu --extra ipynb
# dev deps. are synced by default.

# Suggest user to add alias and source found profile.
echo "Suggested alias: "
echo "alias sve='source .venv/bin/activate'"
echo "Source profile:"
echo "source ${FOUND}"

echo "Script execution complete."