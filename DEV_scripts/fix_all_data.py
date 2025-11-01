#!/usr/bin/env python3
"""
Comprehensive fix for duplicate data parameters in all nexus files
"""

import re

def fix_all_duplicate_data_params(file_path):
    """Fix all duplicate data parameters in a file using better regex"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find all NexusResponse blocks with duplicate data params
    # Pattern looks for the structure:
    # return NexusResponse(
    #     ...
    #     data={"message": ...},
    #     data=some_variable
    # )
    
    def fix_nexus_response(match):
        full_match = match.group(0)
        
        # Extract message and variable parts
        message_match = re.search(r'data=\{"message":\s*([^}]+)\}', full_match)
        var_match = re.search(r'data=([a-zA-Z_]\w*)\s*\)', full_match)
        
        if message_match and var_match:
            message_part = message_match.group(1)
            var_name = var_match.group(1)
            
            # Replace both data lines with merged version
            fixed_match = re.sub(
                r'data=\{"message":\s*[^}]+\},\s*data=[a-zA-Z_]\w*', 
                f'data={{"message": {message_part}, **{var_name}}}',
                full_match
            )
            return fixed_match
        
        return full_match
    
    # Pattern to match entire NexusResponse calls
    nexus_pattern = r'return NexusResponse\s*\([^)]*data=\{"message":[^)]*data=[a-zA-Z_]\w*[^)]*\)'
    content = re.sub(nexus_pattern, fix_nexus_response, content, flags=re.MULTILINE | re.DOTALL)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed duplicate data params in {file_path}")
        return True
    else:
        print(f"ℹ️ No duplicate data params found in {file_path}")
        return False

def main():
    """Fix all nexus files"""
    
    base_dir = r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI"
    
    nexus_files = [
        "AGP/nexus/agp_nexus.py",
        "LOGOS_Agent/nexus/agent_nexus.py", 
        "GUI/nexus/gui_nexus.py"
    ]
    
    for nexus_file in nexus_files:
        file_path = f"{base_dir}\\{nexus_file}"
        try:
            fix_all_duplicate_data_params(file_path)
        except Exception as e:
            print(f"❌ Error fixing {nexus_file}: {e}")

if __name__ == "__main__":
    main()