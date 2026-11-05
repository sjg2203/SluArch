# Changelog

<!--next-version-placeholder-->

## v0.2.0 (05/10/2026)

- Package metadata for standards-compliant build tools and project
- Consolidated the public API
- Added morphology-based slow-oscillation and K-complex detection with original
  sample timing, amplitude measurements, and optional stage/artifact masks
- Added the dual-threshold, cubed-RMS adaptive spindle detector
- Added automatic multimethod spindle detection with slow and
  fast spectral windows, plus a broad-band fallback when peaks are unresolved
- Added unified detector entry points and simplified spindle method names
- Removed the need for YASA and added synthetic signal tests for every detector path
- Dropped end-of-life Python 3.10 and added Python 3.15 support in metadata and CI
- Added test, lint, documentation, and release dependency extras

## v0.1.0 (05/16/2024)

- First release of `SlµArch`!
