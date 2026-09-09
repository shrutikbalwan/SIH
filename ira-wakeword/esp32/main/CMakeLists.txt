idf_component_register(
    SRCS
        "main.cpp"
        "ira_model_data.cc"
        "ira_inference.cpp"
        "ira_features.cpp"
        "ira_audio_guard.cpp"
        "ira_selftest_data.cc"
    INCLUDE_DIRS "."
    REQUIRES esp_timer
)

# The feature frontend relies on IEEE-754 semantics: the mean/variance are
# accumulated in double specifically to avoid the float32 reduction artifact
# documented in ESP32_FEATURE_PARITY_REPORT.md. -ffast-math would licence
# reassociation and break bit-parity with the Python reference.
target_compile_options(${COMPONENT_LIB} PRIVATE -fno-fast-math -O2)
