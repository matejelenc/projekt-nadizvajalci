#!/usr/bin/env bash
set -euo pipefail

MMC_URL=https://file.biolab.si/tmp/uozp-data/mmc.yaml.zip
ALL_COUNTRIES=http://download.geonames.org/export/dump/allCountries.zip
ADMIN1_ASCII_CODES=http://download.geonames.org/export/dump/admin1CodesASCII.txt
COUNTRY_INFO=http://download.geonames.org/export/dump/countryInfo.txt
RAW_DIR=./data/raw

echo "Downloading necessary files..."
mkdir -p "$RAW_DIR"

# Download MMC.
curl -o "${RAW_DIR}/mmc.yaml.zip" "${MMC_URL}"
unzip "${RAW_DIR}/mmc.yaml.zip" -d "${RAW_DIR}"
rm "${RAW_DIR}/mmc.yaml.zip"

# Download GeoNames.
wget -P "${RAW_DIR}" "${ALL_COUNTRIES}"
unzip "${RAW_DIR}/allCountries.zip" -d "${RAW_DIR}"
rm "${RAW_DIR}/allCountries.zip"
wget -P "${RAW_DIR}" "${ADMIN1_ASCII_CODES}"
wget -P "${RAW_DIR}" "${COUNTRY_INFO}"

echo "Download successful, data is stored in ${RAW_DIR}."