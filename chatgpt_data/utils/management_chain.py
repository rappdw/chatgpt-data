#!/usr/bin/env python3
"""
Process Proofpoint.xlsx to produce a dictionary where the key is the value in the "Name" column
with the (employee id) stripped, and the value is an ordered list of the management chain of
that employee up to and including Sumit Dhawan.
"""
from pathlib import Path
import pandas as pd
import re
import json
import pickle


def extract_name_without_id(name_with_id: str) -> str:
    """
    Extract the name part from a string that includes an employee ID in parentheses.
    
    Args:
        name_with_id: String containing name and employee ID, e.g., "John Doe (12345)"
        
    Returns:
        The name without the ID, e.g., "John Doe"
    """
    # Use regex to match everything before the pattern " (digits)"
    match = re.match(r"(.*?)\s+\(\d+\)$", name_with_id)
    if match:
        return match.group(1)
    return name_with_id  # Return original if pattern not found


def build_management_chain(excel_file: Path) -> dict:
    """
    Build a dictionary mapping employee names (without IDs) to their management chain.
    
    Args:
        excel_file: Path to the Proofpoint.xlsx file
        
    Returns:
        Dictionary where keys are employee names without IDs and values are lists
        representing the management chain (ordered from immediate manager to highest level,
        including Sumit Dhawan)
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Create a mapping from unique identifier to name
    id_to_name = dict(zip(df['Unique Identifier'], df['Name']))
    
    # Create a mapping from unique identifier to reports to
    id_to_reports_to = dict(zip(df['Unique Identifier'], df['Reports To']))
    
    # Find Sumit Dhawan's unique identifier and name
    sumit_rows = df[df['Name'].str.contains('Sumit Dhawan', na=False)]
    if sumit_rows.empty:
        raise ValueError("Could not find Sumit Dhawan in the data")
    sumit_id = sumit_rows.iloc[0]['Unique Identifier']
    sumit_name = extract_name_without_id(sumit_rows.iloc[0]['Name'])
    
    # Build the management chain for each employee
    management_chains = {}
    
    for employee_id, employee_name in id_to_name.items():
        # Extract name without ID
        employee_name_clean = extract_name_without_id(employee_name)
        
        # Build the management chain
        chain = []
        current_id = id_to_reports_to.get(employee_id)
        
        while current_id:
            manager_name = id_to_name.get(current_id)
            if manager_name:
                chain.append(extract_name_without_id(manager_name))
            
            # Stop if we've reached Sumit Dhawan
            if current_id == sumit_id:
                break
                
            current_id = id_to_reports_to.get(current_id)
        
        # Store the chain
        management_chains[employee_name_clean] = chain
    
    return management_chains


def save_management_chain(management_chains: dict, output_file: Path, format_type: str = 'json') -> None:
    """
    Save the management chain dictionary to a file.
    
    Args:
        management_chains: Dictionary mapping employee names to management chains
        output_file: Path where the output file should be saved
        format_type: Format to save the data in ('json' or 'pickle')
    """
    if format_type.lower() == 'json':
        with open(output_file, 'w') as f:
            json.dump(management_chains, f, indent=2)
    elif format_type.lower() == 'pickle':
        with open(output_file, 'wb') as f:
            pickle.dump(management_chains, f)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def main():
    """Entry point for the console script."""
    data_dir = Path(__file__).parent.parent.parent / 'rawdata'
    proofpoint_file = data_dir / 'Proofpoint.xlsx'
    
    # Build the management chain
    management_chains = build_management_chain(proofpoint_file)
    
    # Save as JSON (easily readable)
    json_output = data_dir / 'management_chains.json'
    save_management_chain(management_chains, json_output, 'json')
    
    # Save as pickle (for efficient loading in Python)
    pickle_output = data_dir / 'management_chains.pkl'
    save_management_chain(management_chains, pickle_output, 'pickle')
    
    print(f"Management chains saved to:")
    print(f"  - JSON: {json_output}")
    print(f"  - Pickle: {pickle_output}")


if __name__ == '__main__':
    main()
