#!/bin/bash

# Download the entire latest CATH release folder from FTP server
set -e

# CATH FTP server details
FTP_SERVER="orengoftp.biochem.ucl.ac.uk"
FTP_PATH="/cath/releases/latest-release"

# Create data directory
mkdir -p data

echo "Downloading CATH latest release data..."

# Download release metadata
echo "Fetching release metadata..."
if curl -f -s -S https://api.cathdb.info/api/v0.6/releases/latest -o data/cath_latest_release.json; then
    echo "✓ Metadata fetched successfully"
    if command -v jq &> /dev/null; then
        RELEASE_VERSION=$(jq -r '.version' data/cath_latest_release.json 2>/dev/null || echo "unknown")
        echo "Release version: $RELEASE_VERSION"
    fi
else
    echo "✗ Failed to fetch metadata (continuing with FTP download)"
fi

# Download files using wget
echo "Downloading files from FTP server..."
cd data

if ! command -v wget &> /dev/null; then
    echo "✗ ERROR: wget is not installed"
    echo "Install with: brew install wget (macOS) or apt-get install wget (Linux)"
    exit 1
fi

if wget -r -nH --cut-dirs=3 -np \
     -R "index.html*,cath-superfamily-seqs-*.fa" \
     --reject-regex '.*\?.*' \
     -X "sequence-data/sequence-by-superfamily" \
     -q --show-progress \
     "ftp://${FTP_SERVER}${FTP_PATH}/"; then
    echo "✓ Download complete"
else
    echo "✗ ERROR: Download failed"
    exit 1
fi

cd ..

echo "Files saved to: $(pwd)/data/"
du -sh data/* 2>/dev/null | head -10 || echo "No files found"
