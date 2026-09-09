# ESP-IDF top-level project file for the IRA wake-word firmware.
#
# Expected ESP-IDF: v5.1.x or newer (developed against v5.1.4)
# Target          : esp32s3
#
#   idf.py set-target esp32s3
#   idf.py build
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ira_wakeword)
