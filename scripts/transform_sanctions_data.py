
"""
UK Financial Sanctions Data Processing Script

This script processes the UK financial sanctions list from:
https://www.gov.uk/government/publications/financial-sanctions-consolidated-list-of-targets

The script performs data cleaning, transformation and consolidation to create
a structured dataset suitable for comparison with customer records.
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# Constants
INPUT_FILE = os.path.join( 'data', 'UK_Sanctions_List.xlsx')
OUTPUT_FILE = os.path.join( 'output', 'UK_Sanctions_Processed.csv')

def load_raw_data(filepath):
    """
    Load raw data from Excel file
    
    Args:
        filepath (str): Path to input Excel file
        
    Returns:
        pd.DataFrame: Raw sanctions data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath, header=1)
    df.sort_values(by='Group ID', ascending=True, inplace=True)
    return df

def clean_and_transform(df):
    """
    Perform data cleaning and transformation
    
    Args:
        df (pd.DataFrame): Raw sanctions data
        
    Returns:
        pd.DataFrame: Transformed and cleaned data
    """
    print("Performing data cleaning and transformation...")
    # Country corrections
    country_corrections = {
        'UK': 'United Kingdom',
        'United States of America': 'United States',
        'DPRK': 'North Korea',
        'Syrian Arab Republic': 'Syria',
        'Russian Federation (as at November 2010)': 'Russia',
        'RUSSIA': 'Russia',
        'Iraq (previous address)': 'Iraq',
        'Occupied Palestinian Territories': 'Palestinian Territories',
        'Gaza': 'Palestinian Territories',
        'Türkiye': 'Turkey',
        'Congo (Democratic Republic)': 'Democratic Republic of the Congo',
        'Moscow': 'Russia',
        '115088': np.nan
    }
    # Apply corrections
    df['Country'] = df['Country'].replace(country_corrections)
    # Remove clearly invalid entries
    invalid_countries = ['115088', 'Moscow', None, np.nan]
    df['Country'] = df['Country'].where(~df['Country'].isin(invalid_countries), np.nan)

    # Remove short or non-string values
    df['Country'] = df['Country'].apply(lambda x: x if isinstance(x, str) and len(x.strip()) > 2 else np.nan)
    
    # Convert date columns
    df['DOB'] = df['DOB'].replace(r'00/00/\d{4}', '01/01/1900', regex=True)
    df['DOB'] = pd.to_datetime(df['DOB'], format='%d/%m/%Y', errors='coerce')
    df['Listed On'] = pd.to_datetime(df['Listed On'], format='%d/%m/%Y')
    df['UK Sanctions List Date Designated'] = pd.to_datetime(
        df['UK Sanctions List Date Designated'], format='%d/%m/%Y')
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], format='%d/%m/%Y')
    
    # Combine name and address components
    df['full_name'] = df[['Name 6', 'Name 1', 'Name 2', 'Name 3', 'Name 4', 'Name 5']]\
        .fillna('').agg(' '.join, axis=1).str.strip()
    df['Full Address'] = df[['Address 1', 'Address 2', 'Address 3', 
                            'Address 4', 'Address 5', 'Address 6']]\
        .fillna('').agg(' '.join, axis=1).str.strip()
    
    # Standardize name formatting
    df['full_name'] = df['full_name'].str.title()
    df['Full Address'] = df['Full Address'].replace('', np.nan)
    
    return df

def clean_nationalities(nat):
    """Clean and split nationality field"""
    if pd.isna(nat):
        return []
    
    # Reemplazar puntos, comas y dividir por delimitadores
    nat = nat.replace(';', '.').replace(',', '.')
    
    # Extraer posibles países, eliminando números y paréntesis como (1), (2), etc.
    parts = re.split(r"\(?\d+\)?\.?", nat)
    
    # Limpiar cada parte: quitar espacios, puntos finales, etc.
    cleaned = [part.strip().rstrip('.') for part in parts if part.strip()]
    
    return cleaned


def get_primary_name(group):
    """Extract primary name from group, handling duplicates and missing values"""
    
    # Filtrar el "Primary name" según el Alias Type
    primary_row = group[group['Alias Type'] == 'Primary name']
    
    # Si hay un "Primary name" claro, lo devolvemos
    if not primary_row.empty:
        return primary_row['full_name'].iloc[0]
    
    # Si no hay "Primary name", tomamos el primer full_name no nulo
    non_null_names = group['full_name'].dropna()
    if not non_null_names.empty:
        return non_null_names.iloc[0]
    
    # Si no hay nombres disponibles, devolvemos None
    return None

def get_aliases(group):
    """Extract all aliases from group"""
    names = group['full_name'].dropna().unique().tolist()
    primary = get_primary_name(group)
    return [name for name in names]

def get_addresses(group):
    """Extract all unique addresses from group"""
    return group['Full Address'].dropna().unique().tolist()

def get_related_countries(group):
    """Extract related countries from Country and Nationality fields"""
    # Extraer países de la columna 'Country'
    countries = group['Country'].dropna().unique().tolist()
    
    # Si no hay países, intentamos usar la columna 'Nationality' (limpiada)
    if not countries and 'Nationality' in group:
        # Limpiar la nacionalidad antes de agregarla
        nationality = group['Nationality'].dropna().unique().tolist()
        nationality_cleaned = [clean_nationalities(nat) for nat in nationality]
        # Aplanar la lista de listas y eliminar duplicados
        return list(set([item for sublist in nationality_cleaned for item in sublist]))
    
    return countries

def get_other_info(group):
    """Extract other information from group"""
    return group['Other Information'].iloc[0] if 'Other Information' in group else None

def get_nationality(group):
    """Extract nationality from group"""
    return group['Nationality'].iloc[0] if 'Nationality' in group else None


def get_regime(group):
    """Extract regime from group"""
    return group['Regime'].iloc[0] if 'Regime' in group else None

def get_dates(group):
    """Extract all relevant dates from group"""
    return {
        'DOB': group['DOB'].dropna().iloc[0] if not group['DOB'].isna().all() else None,
        'listed_on': group['Listed On'].dropna().iloc[0] if not group['Listed On'].isna().all() else None,
        'uk_sanctions_date': group['UK Sanctions List Date Designated'].dropna().iloc[0] 
            if not group['UK Sanctions List Date Designated'].isna().all() else None,
        'last_updated': group['Last Updated'].dropna().iloc[0] 
            if not group['Last Updated'].isna().all() else None
    }

def consolidate_data(df):
    """
    Consolidate data by Group ID
    
    Args:
        df (pd.DataFrame): Cleaned data
        
    Returns:
        pd.DataFrame: Consolidated sanctions data
    """
    print("Consolidating data by Group ID...")
    
    grouped_df = df.groupby('Group ID').apply(lambda group: pd.Series({
        'group_id': group.name,
        'group_type': group['Group Type'].dropna().iloc[0] 
            if not group['Group Type'].isna().all() else pd.NA,
        'primary_name': get_primary_name(group),
        'aliases': get_aliases(group),
        'addresses': get_addresses(group),
        'regime': get_regime(group),
        'related_countries': get_related_countries(group),
        'nationality': get_nationality(group),
        'other_information': get_other_info(group),
        'dates': get_dates(group)
    })).reset_index(drop=True)
    
    # Expand dates into separate columns
    dates_df = pd.json_normalize(grouped_df['dates'])
    final_df = pd.concat([grouped_df.drop(columns='dates'), dates_df], axis=1)
    
    return final_df

def analyze_data_quality(df):
    """
    Analyze and report data quality issues and print duplicates
    
    Args:
        df (pd.DataFrame): Processed data
        
    Returns:
        dict: Data quality metrics
    """
    print("Analyzing data quality...")
    
    # Detect duplicates
    duplicated_rows = df[df['primary_name'].duplicated(keep=False)]  # keep=False keeps all duplicates not just the first
    
    # Print duplicates
    if not duplicated_rows.empty:
        print("\nDuplicated Entries:")
        print(duplicated_rows)  # Imprime todas las filas duplicadas
    
    quality_report = {
        'total_records': len(df),
        'unique_entities': df['group_id'].nunique(),
        'missing_dob': df['DOB'].isna().sum(),
        'missing_addresses': df['addresses'].apply(len).eq(0).sum(),
        'missing_nationality': df['nationality'].isna().sum(),
        'missing_related_countries': df['related_countries'].apply(len).eq(0).sum(),
        'duplicate_names': df['primary_name'].duplicated().sum(),
        'missing_group_type': df['group_type'].isna().sum(),
        'missing_primary_name': df['primary_name'].isna().sum(),
        'missing_aliases': df['aliases'].apply(len).eq(0).sum(),
        'missing_regime': df['regime'].isna().sum(),
        'missing_other_information': df['other_information'].isna().sum()
    }
    
    return quality_report


def export_results(df, output_path, quality_report):
    """
    Export processed data and quality report
    
    Args:
        df (pd.DataFrame): Processed data
        output_path (str): Output file path
        quality_report (dict): Data quality metrics
    """
    print(f"Exporting results to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export processed data
    df.to_csv(output_path, index=False)
    
    # Print quality report
    print("\nData Quality Report:")
    for metric, value in quality_report.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")

def main():
    """Main execution function"""
    try:        
        # Load and process data
        raw_df = load_raw_data(INPUT_FILE)
        cleaned_df = clean_and_transform(raw_df)
        consolidated_df = consolidate_data(cleaned_df)
        
        # Analyze and export
        quality_report = analyze_data_quality(consolidated_df)
        export_results(consolidated_df, OUTPUT_FILE, quality_report)
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == '__main__':
    main()