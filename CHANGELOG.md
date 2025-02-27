# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `get_usage_data` CLI tool for fetching data via the Enterprise Compliance API
- Direct integration with OpenAI's Enterprise Compliance API
- Option to automatically run analysis after fetching new data
- Support for API key and organization ID via environment variables

### Changed
- Enhanced user engagement report to show active periods as a percentage of eligible periods since user creation
- Improved report formatting: removed account_id and public_id columns, consolidated name/email into a single display_name column
- Updated data viewer (index.html) to load actual CSV data from files instead of using sample data
- Changed data viewer title to "ChatGPT Enterprise Engagement Metrics"

### Fixed
- Fixed issue with `avg_messages` column in user engagement report being interpreted as text in Excel
- Fixed issue where active period percentages could incorrectly exceed 100%
- Fixed duplicate columns in user engagement report

## [1.0.0] - 2025-02-25

### Added
- New comprehensive `all_trends` CLI tool that combines user and GPT trend analyses
- Command line options to control which analyses and reports to generate
- Detailed documentation in README.md for all CLI tools
- Enhanced test coverage for all components

### Changed
- Improved error handling for data loading
- Better organization of code with modular architecture
- Enhanced reporting capabilities with optional filtering

### Fixed
- Proper handling of different CSV file encodings
- Accurate filtering of pending users in reports
- Consistent handling of date periods across all analyses
