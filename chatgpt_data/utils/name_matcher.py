"""Utilities for name normalization and matching."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rapidfuzz import fuzz


class NameMatcher:
    """Class for normalizing and matching human names across different formats and variations.
    
    The NameMatcher provides robust name matching capabilities to handle common challenges
    in human name comparisons, including:
    
    - Case and whitespace normalization
    - Common nickname variations (e.g., "Robert" -> "Bob", "William" -> "Bill")
    - Handling of middle/maiden names and their variations
    - Fuzzy matching for names with slight spelling differences
    - Management chain resolution for organizational hierarchies
    
    This class is particularly useful for:
    1. Matching display names against formal names in organizational data
    2. Resolving management chains when names may be formatted differently
    3. Normalizing names for consistent representation in reports
    4. Finding employees in hierarchical structures with partial or variant name information
    
    The matching process follows this sequence:
    1. Try exact match against known names
    2. Try normalized name match (lowercase, whitespace normalized)
    3. Try component matching (first+last name for complex names)
    4. Try matching against expanded maiden name variations
    5. Fall back to fuzzy matching with configurable threshold
    
    Usage:
        # Initialize with management chain data
        matcher = NameMatcher(management_chains)
        
        # Get normalized version of a name
        normalized = matcher.normalize_name("John A. Smith")
        
        # Match a name variant to a canonical name
        matched = matcher.match_name("Johnny Smith")
        
        # Get management chain for an employee
        chain = matcher.get_management_chain("Bob Johnson")
    """
    
    def __init__(self, management_chains: Optional[Dict] = None, config_path: Optional[str] = None):
        """Initialize the NameMatcher.
        
        Args:
            management_chains: Dictionary of management chains to build normalized name mapping
            config_path: Path to the name matches config file (defaults to '.name_matches' in current directory)
        """
        self.management_chains = management_chains
        self.normalized_names = {}
        self.name_matches = {}
        
        # Load name matches from config file
        self.config_path = config_path or '.name_matches'
        self._load_name_matches()
        
        # Build normalized name mapping if management chains are provided
        if self.management_chains:
            self._build_normalized_name_mapping()
    
    def _load_name_matches(self) -> None:
        """Load name matches from the config file.
        
        The config file should be a JSON file with a dictionary mapping from
        display names to canonical names in the management chain.
        """
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.name_matches = json.load(f)
                logging.info(f"Loaded {len(self.name_matches)} name matches from {self.config_path}")
            else:
                logging.info(f"Name matches config file {self.config_path} not found. Using empty mapping.")
        except Exception as e:
            logging.warning(f"Error loading name matches config: {e}. Using empty mapping.")
            self.name_matches = {}
    
    def _build_normalized_name_mapping(self) -> None:
        """Build a mapping of normalized names to original names."""
        for name in self.management_chains.keys():
            # Normalize the name (lowercase, remove extra spaces)
            normalized = self.normalize_name(name)
            self.normalized_names[normalized] = name
            
            # Also add common name variations (e.g., Dan -> Daniel)
            variations = self.get_name_variations(name)
            for variation in variations:
                normalized_variation = self.normalize_name(variation)
                if normalized_variation != normalized:
                    self.normalized_names[normalized_variation] = name
            
            # Add first name + last name version for names with 3+ parts (handling maiden names)
            name_components = self.extract_name_components(name)
            if len(normalized.split()) >= 3 and "first_name" in name_components and "last_name" in name_components:
                simplified_name = f"{name_components['first_name']} {name_components['last_name']}"
                if simplified_name != normalized:
                    self.normalized_names[simplified_name] = name
            
            # For names with exactly 2 parts, add potential maiden name variations
            if len(normalized.split()) == 2:
                maiden_variations = self.expand_with_common_maiden_names(name)
                for variation in maiden_variations:
                    self.normalized_names[variation] = name
    
    def normalize_name(self, name: str) -> str:
        """Normalize a name for comparison.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name (lowercase, no extra spaces, no parenthetical text)
        """
        if not name or pd.isna(name):
            return ""
            
        # Remove parenthetical text (like foreign scripts or additional information)
        name_without_parentheses = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', name_without_parentheses.lower().strip())
        return normalized
    
    def extract_name_components(self, name: str) -> dict:
        """Extract components from a name for more flexible matching.
        
        Args:
            name: Full name to extract components from
            
        Returns:
            Dictionary with first_name, last_name, and full_name keys
        """
        if not name or pd.isna(name):
            return {"first_name": "", "last_name": "", "full_name": ""}
            
        # Normalize the name first
        normalized = self.normalize_name(name)
        if not normalized:
            return {"first_name": "", "last_name": "", "full_name": normalized}
            
        # Split the name into parts
        parts = normalized.split()
        
        result = {
            "full_name": normalized,
            "first_name": parts[0] if parts else "",
            "last_name": parts[-1] if parts else ""
        }
        
        # If there are at least 3 parts, store potential middle/maiden names
        if len(parts) >= 3:
            result["middle_names"] = parts[1:-1]
        
        return result
    
    def expand_with_common_maiden_names(self, name: str) -> List[str]:
        """Generate potential variations by adding common maiden name positions.
        
        This helps match a first+last name against a first+maiden+last name.
        
        Args:
            name: Original name
            
        Returns:
            List of name variations with potential maiden name placements
        """
        variations = []
        
        # Skip if name is empty or not a string
        if not name or pd.isna(name):
            return variations
            
        # Split the name into parts
        parts = self.normalize_name(name).split()
        
        # We only handle the case of exactly 2 parts (first + last)
        if len(parts) != 2:
            return variations
            
        # Common maiden name positions (typically between first and last name)
        # We'll use some common placeholder maiden names
        common_maiden_placeholders = ["middlename", "maidenname"]
        
        for placeholder in common_maiden_placeholders:
            # Create a variation with the placeholder in the middle
            variation = f"{parts[0]} {placeholder} {parts[1]}"
            variations.append(variation)
            
        return variations
    
    def get_name_variations(self, name: str) -> List[str]:
        """Generate common variations of a name.
        
        Args:
            name: Original name
            
        Returns:
            List of name variations
        """
        variations = []
        
        # Skip if name is empty or not a string
        if not name or pd.isna(name):
            return variations
            
        # Split the name into parts
        parts = name.split()
        if len(parts) < 1:
            return variations
            
        # Common first name variations
        first_name = parts[0]
        common_variations = {
            "nathan": ["nate"],
            "nate": ["nathan"],
            "daniel": ["dan", "danny"],
            "dan": ["daniel", "danny"],
            "danny": ["daniel", "dan"],
            "michael": ["mike", "mick"],
            "mike": ["michael", "mick"],
            "robert": ["rob", "bob", "bobby"],
            "rob": ["robert", "bob", "bobby"],
            "bob": ["robert", "rob", "bobby"],
            "william": ["will", "bill", "billy"],
            "will": ["william", "bill", "billy"],
            "bill": ["william", "will", "billy"],
            "richard": ["rick", "dick", "rich"],
            "rick": ["richard", "dick", "rich"],
            "james": ["jim", "jimmy"],
            "jim": ["james", "jimmy"],
            "thomas": ["tom", "tommy"],
            "tom": ["thomas", "tommy"],
            "john": ["johnny", "jon"],
            "jonathan": ["jon", "jonny"],
            "christopher": ["chris", "topher"],
            "chris": ["christopher", "topher"],
            "joseph": ["joe", "joey"],
            "joe": ["joseph", "joey"],
            "david": ["dave", "davey"],
            "dave": ["david", "davey"],
            "charles": ["chuck", "charlie"],
            "chuck": ["charles", "charlie"],
            "charlie": ["charles", "chuck"],
            "matthew": ["matt", "matty"],
            "matt": ["matthew", "matty"],
            "nicholas": ["nick", "nicky"],
            "nick": ["nicholas", "nicky"],
            "anthony": ["tony", "ant"],
            "tony": ["anthony", "ant"],
            "steven": ["steve", "stevie"],
            "steve": ["steven", "stevie"],
            "andrew": ["andy", "drew"],
            "andy": ["andrew", "drew"],
            "drew": ["andrew", "andy"],
            "jennifer": ["jen", "jenny"],
            "jen": ["jennifer", "jenny"],
            "jessica": ["jess", "jessie"],
            "jess": ["jessica", "jessie"],
            "elizabeth": ["liz", "beth", "eliza"],
            "liz": ["elizabeth", "beth", "eliza"],
            "beth": ["elizabeth", "liz", "eliza"],
            "katherine": ["kate", "katie", "kathy"],
            "kate": ["katherine", "katie", "kathy"],
            "katie": ["katherine", "kate", "kathy"],
            "kathy": ["katherine", "kate", "katie"],
            "margaret": ["maggie", "meg", "peggy"],
            "maggie": ["margaret", "meg", "peggy"],
            "patricia": ["pat", "patty", "tricia"],
            "pat": ["patricia", "patty", "tricia"],
            "stephanie": ["steph", "stephie"],
            "steph": ["stephanie", "stephie"],
            "constanza": ["connie", "constance"],
            "constance": ["connie", "constanza"],
            "tim": ["timothy"],
            "timothy": ["tim"],
            "benjamin": ["ben"],
            "ben": ["benjamin"],
        }
        
        first_lower = first_name.lower()
        if first_lower in common_variations:
            for variation in common_variations[first_lower]:
                if len(parts) > 1:
                    # Create full name with the variation
                    variations.append(f"{variation.capitalize()} {' '.join(parts[1:])}")
                else:
                    variations.append(variation.capitalize())
                    
        return variations
    
    def match_name(self, name: str, threshold: int = 85) -> Optional[str]:
        """Match a name against the normalized name mapping.
        
        Args:
            name: Name to match
            threshold: Minimum fuzzy match score threshold
            
        Returns:
            Matched original name if found, None otherwise
        """
        if not self.management_chains or not name or pd.isna(name):
            return None
        
        # Try exact match first
        if name in self.management_chains:
            return name
            
        # Try normalized name match
        normalized_name = self.normalize_name(name)
        if normalized_name in self.normalized_names:
            return self.normalized_names[normalized_name]
        
        # Extract name components for more flexible matching
        name_components = self.extract_name_components(name)
        
        # Try to match based on first name + last name if we have a complex name
        if len(normalized_name.split()) >= 3 and "first_name" in name_components and "last_name" in name_components:
            simplified_name = f"{name_components['first_name']} {name_components['last_name']}"
            if simplified_name in self.normalized_names:
                return self.normalized_names[simplified_name]
        
        # If we have a simple name (first + last), try matching against expanded maiden name variations
        if len(normalized_name.split()) == 2:
            maiden_variations = self.expand_with_common_maiden_names(name)
            for variation in maiden_variations:
                if variation in self.normalized_names:
                    return self.normalized_names[variation]
        
        # Try fuzzy matching if no exact or normalized match
        best_match = None
        best_score = 0
        
        for key_name in self.management_chains.keys():
            # Try exact component matching first (first name + last name)
            key_components = self.extract_name_components(key_name)
            
            # Check if first and last names match exactly
            if (name_components["first_name"] == key_components["first_name"] and 
                name_components["last_name"] == key_components["last_name"]):
                return key_name
            
            # If that fails, try fuzzy matching on the full name
            score = fuzz.ratio(normalized_name, self.normalize_name(key_name))
            if score > threshold and score > best_score:
                best_score = score
                best_match = key_name
                
        return best_match
    
    def get_management_chain(self, name: str) -> List[str]:
        """Get the management chain for an employee.
        
        Args:
            name: Employee name
            
        Returns:
            List of managers in the chain (in reverse order, from employee to CEO)
        """
        if not self.management_chains or name is None or pd.isna(name):
            return []
        
        # Try to match the name
        matched_name = self.match_name(name)
        
        # Return the management chain if a match was found
        if matched_name and matched_name in self.management_chains:
            return self.management_chains[matched_name]
        
        # Check if the name is in our custom name matches dictionary
        if name in self.name_matches:
            canonical_name = self.name_matches[name]
            if canonical_name in self.management_chains:
                logging.info(f"Found management chain for {name} using custom name mapping to {canonical_name}")
                return self.management_chains[canonical_name]
        
        # No management chain found
        logging.warning(f"No management chain found for {name}")
        return []