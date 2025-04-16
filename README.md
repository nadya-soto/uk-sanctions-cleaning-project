# UK Financial Sanctions Data Processing

## Overview
This project processes the UK financial sanctions list to create a structured dataset suitable for comparing against customer records.

## Requirements
- Python 3.7+
- pandas
- numpy

## Installation
1. Clone this repository
2. Install dependencies:
pip install pandas numpy openpyxl

## Usage
1. Place the raw UK sanctions data file (Excel format) in the `data/` directory
2. Run the processing script:

python process_sanctions_data.py

3. The processed data will be saved to `output/UK_Sanctions_Processed.csv`

## Data Quality Notes
Key data quality issues identified:
    Missing Date of Birth: 1782 records (approximately 38% of total records)
    Missing Addresses: 2487 records (approximately 53% of total records)
    Missing Nationality: 2092 records (approximately 45% of total records)
    Missing Related Countries: 733 records (approximately 16% of total records)
    Duplicate Primary Names: 3 duplicate records identified
    No missing fields for Group Type, Primary Name, Aliases, Regime, or Other Information.

## Output Fields
The processed dataset includes:
group_id: Unique identifier for the entity.
group_type: Type of the entity (e.g., individual, entity, etc.).
primary_name: The primary name of the entity.
aliases: Other names associated with the entity.
addresses: Addresses associated with the entity.
regime: Any applicable regime for the sanctions.
related_countries: Countries related to the entity (e.g., nationality or country of residence).
nationality: The nationality of the entity.
other_information: Additional information related to the entity.
dates: Key dates including date of birth (DOB), listing date, sanctions designation date, and last updated date.