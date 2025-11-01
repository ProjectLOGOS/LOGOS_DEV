#!/usr/bin/env python3
"""
Quick fix to merge duplicate data parameters in NexusResponse calls
"""

import re

def fix_duplicate_data_params(file_path):
    """Fix duplicate data parameters in a file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find duplicate data parameters
    # Match: data={"message": "..."}, followed by data=some_var
    pattern = r'data=\{"message":\s*([^}]+)\},\s*data=([a-zA-Z_]\w*)'
    
    def merge_data_params(match):
        message_part = match.group(1)
        var_name = match.group(2)
        # Merge into single data dict
        return f'data={{"message": {message_part}, **{var_name}}}'
    
    new_content = re.sub(pattern, merge_data_params, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed duplicate data params in {file_path}")
        return True
    return False

# Fix the AGP nexus file
agp_file = r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI\AGP\nexus\agp_nexus.py"
logos_agent_file = r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI\LOGOS_Agent\nexus\agent_nexus.py"
gui_file = r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI\GUI\nexus\gui_nexus.py"

fix_duplicate_data_params(agp_file)
fix_duplicate_data_params(logos_agent_file)
fix_duplicate_data_params(gui_file)