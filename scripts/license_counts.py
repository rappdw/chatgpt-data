#!/usr/bin/env python3
import pandas as pd
import sys
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import argparse
import csv

def read_employee_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read employee data from CSV or Excel file.
    
    Args:
        file_path: Path to the employee file (CSV or Excel)
        
    Returns:
        DataFrame containing employee data or None if an error occurred
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        # Validate required columns
        required_cols = ["Unique Identifier", "Reports To"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {', '.join(missing_cols)}")
            return None
            
        return df
    except Exception as e:
        print(f"Error reading employee file: {e}")
        return None

def read_leader_ids(file_path: str) -> Optional[Set[str]]:
    """
    Read leader IDs from a text file.
    
    Args:
        file_path: Path to the leader ID file
        
    Returns:
        Set of leader IDs or None if an error occurred
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: Leader ID file not found: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            leader_ids = {line.strip() for line in f if line.strip()}
            
        print(f"Read {len(leader_ids)} leader IDs")
        return leader_ids
    except Exception as e:
        print(f"Error reading leader ID file: {e}")
        return None

def build_org_hierarchy(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], str]:
    """
    Build organization hierarchy from employee data.
    
    Args:
        df: DataFrame containing employee data
        
    Returns:
        Tuple containing:
        - Dictionary mapping each employee to their direct reports
        - CEO's unique identifier (employee with no manager)
    """
    hierarchy = defaultdict(list)
    all_employees = set(df["Unique Identifier"])
    
    # Find employees who are listed as managers
    for _, row in df.iterrows():
        manager_id = row["Reports To"]
        employee_id = row["Unique Identifier"]
        
        if pd.notna(manager_id):
            hierarchy[manager_id].append(employee_id)
    
    # Identify the CEO (employee with null "Reports To" value)
    ceo_candidates = df[df["Reports To"].isna()]["Unique Identifier"].tolist()
    
    if not ceo_candidates:
        print("Warning: Could not identify a CEO (no employee without a manager)")
        # Use a placeholder or the first employee as CEO
        return hierarchy, df["Unique Identifier"].iloc[0]
    
    if len(ceo_candidates) > 1:
        print(f"Warning: Multiple employees don't have managers: {ceo_candidates}")
        print(f"Using {ceo_candidates[0]} as the CEO")
    
    ceo_id = ceo_candidates[0]
    print(f"CEO identified as: {ceo_id}")
    
    # Verify the CEO exists in the employee list
    if ceo_id not in all_employees:
        print(f"Warning: CEO ID {ceo_id} not found in employee list")
    
    return hierarchy, ceo_id

def count_engineers_under_leader(
    leader_id: str, 
    hierarchy: Dict[str, List[str]], 
    engineering_leaders: Set[str]
) -> int:
    """
    Count the number of engineers reporting up through a leader.
    
    Args:
        leader_id: Unique identifier of the leader
        hierarchy: Organization hierarchy
        engineering_leaders: Set of engineering leader IDs
        
    Returns:
        Count of engineers reporting to the leader
    """
    # Base case: if leader is an engineering leader, count them and all reports
    if leader_id in engineering_leaders:
        # Count all employees in this subtree
        return count_all_employees_under_leader(leader_id, hierarchy)
    
    # Recursive case: sum up engineers from direct reports
    engineer_count = 0
    for direct_report in hierarchy.get(leader_id, []):
        engineer_count += count_engineers_under_leader(direct_report, hierarchy, engineering_leaders)
    
    return engineer_count

def count_all_employees_under_leader(leader_id: str, hierarchy: Dict[str, List[str]]) -> int:
    """
    Count the total number of employees under a leader (including the leader).
    
    Args:
        leader_id: Unique identifier of the leader
        hierarchy: Organization hierarchy
        
    Returns:
        Count of employees under the leader (including the leader)
    """
    # Start with 1 for the leader themselves
    count = 1
    
    # Add direct reports and their reports recursively
    for direct_report in hierarchy.get(leader_id, []):
        count += count_all_employees_under_leader(direct_report, hierarchy)
    
    return count

def analyze_org_hierarchy(employee_file: str, leader_id_file: str) -> Optional[pd.DataFrame]:
    """
    Analyze the organizational hierarchy to determine how many engineers
    report to each direct report of the CEO.
    
    Args:
        employee_file: Path to the employee file (CSV or Excel)
        leader_id_file: Path to the leader ID file
        
    Returns:
        DataFrame containing analysis results or None if an error occurred
    """
    # Read input files
    employees_df = read_employee_file(employee_file)
    if employees_df is None:
        return None
        
    engineering_leaders = read_leader_ids(leader_id_file)
    if engineering_leaders is None:
        return None
    
    # Build organization hierarchy
    hierarchy, ceo_id = build_org_hierarchy(employees_df)
    print(f"Identified CEO ID: {ceo_id}")
    
    # Get CEO's direct reports
    ceo_direct_reports = hierarchy.get(ceo_id, [])
    if not ceo_direct_reports:
        print("Warning: CEO has no direct reports")
        return pd.DataFrame(columns=["Name", "Unique Identifier", "# Engineers Reporting Up"])
    
    # For each direct report, compute engineers reporting up
    results = []
    total_engineers = 0
    
    # Find the CEO in the employee data
    ceo_record = employees_df[employees_df["Unique Identifier"] == ceo_id].iloc[0]
    ceo_name = ceo_record.get("Name", f"Employee {ceo_id}")
    
    for direct_report_id in ceo_direct_reports:
        # Skip if this is somehow the CEO (avoid duplicates)
        if direct_report_id == ceo_id:
            continue
            
        # Find employee record
        employee_record = employees_df[employees_df["Unique Identifier"] == direct_report_id].iloc[0]
        
        # Get name (if available)
        name = employee_record.get("Name", f"Employee {direct_report_id}")
        
        # Count engineers
        engineer_count = count_engineers_under_leader(direct_report_id, hierarchy, engineering_leaders)
        total_engineers += engineer_count
        
        # Build result record
        result = {
            "Name": name,
            "Unique Identifier": direct_report_id,
            "# Engineers Reporting Up": engineer_count
        }
        
        # Add optional fields if available
        for field in ["Line Detail 1", "Organization Name"]:
            if field in employee_record:
                result[field] = employee_record[field]
        
        results.append(result)
    
    # Add CEO to the results with the total count
    ceo_result = {
        "Name": ceo_name,
        "Unique Identifier": ceo_id,
        "# Engineers Reporting Up": total_engineers  # CEO gets the total count
    }
    
    # Add optional fields for CEO if available
    for field in ["Line Detail 1", "Organization Name"]:
        if field in ceo_record:
            ceo_result[field] = ceo_record[field]
    
    # Add CEO to results
    results.append(ceo_result)
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Analyze organizational hierarchy")
    parser.add_argument("--employee-file", "-e", required=True,
                        help="Path to employee file (CSV or Excel)")
    parser.add_argument("--leader-file", "-l", required=True,
                        help="Path to engineering leader ID file")
    parser.add_argument("--output", "-o",
                        help="Output file path (CSV). If not provided, results will be printed.")
    parser.add_argument("--license-cost", "-c", type=float, default=720.0,
                        help="Annual license cost per engineer (default: $720, representing $60/month)")
    
    args = parser.parse_args()
    
    # Run analysis
    result_df = analyze_org_hierarchy(args.employee_file, args.leader_file)
    
    if result_df is not None:
        # Display full results for console output
        print("\nResults:")
        print(result_df.to_string(index=False))
        
        # Verify data has engineer counts
        total_engineers = result_df["# Engineers Reporting Up"].sum() // 2  # Adjust for double counting
        total_cost = total_engineers * args.license_cost
        print(f"\nTotal engineers across all organizations: {total_engineers:,}")
        print(f"Total annual license cost: ${total_cost:,.2f}")
        
        # Save results if output file specified
        if args.output:
            try:
                # Create a simplified DataFrame with just the name and count
                simplified_df = result_df[["Name", "# Engineers Reporting Up"]].copy()
                
                # Rename column for clarity
                simplified_df = simplified_df.rename(columns={"# Engineers Reporting Up": "Engineering License Count"})
                
                # Ensure data types are preserved
                simplified_df["Engineering License Count"] = simplified_df["Engineering License Count"].astype(int)
                
                # Add annual cost column
                simplified_df["Annual Cost"] = simplified_df["Engineering License Count"] * args.license_cost
                
                # Filter out duplicate entries and keep the entry with the highest count for each name
                simplified_df = simplified_df.sort_values("Engineering License Count", ascending=False).drop_duplicates(subset=["Name"], keep="first")
                
                # Format the values for better readability
                simplified_df["Engineering License Count"] = simplified_df["Engineering License Count"].map(lambda x: f"{x:,}")
                simplified_df["Annual Cost"] = simplified_df["Annual Cost"].map(lambda x: f"${x:,.0f}")
                
                # Save simplified data to CSV
                simplified_df.to_csv(args.output, index=False)
                print(f"\nSimplified results saved to {args.output}")
                
            except Exception as e:
                print(f"Error saving results to {args.output}: {e}")
                
if __name__ == "__main__":
    main()
