#!/usr/bin/env python3
"""
Quick fix to replace all 'message=' parameters in NexusResponse with proper data fields
"""

import os
import re

def fix_nexus_response_file(file_path):
    """Fix all NexusResponse calls in a single file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find NexusResponse calls with message parameter
    # This pattern looks for NexusResponse( ... message="..." ... )
    pattern = r'(NexusResponse\s*\(\s*[^)]*?)message=([^,)]+)(.*?\))'
    
    def replace_message_with_data(match):
        before = match.group(1)
        message_value = match.group(2)
        after = match.group(3)
        
        # Check if there's already a data parameter
        if 'data=' in before or 'data=' in after:
            # If data exists, we need to add the message to the existing data dict
            # For now, just convert to error if success=False, otherwise skip
            if 'success=False' in before:
                return f'{before}error={message_value}{after}'
            else:
                # Add message to data - this needs more complex parsing
                # For now, just convert to a data dict
                return f'{before}data={{"message": {message_value}}}{after}'
        else:
            # No data parameter, create one with the message
            return f'{before}data={{"message": {message_value}}}{after}'
    
    # Apply the replacement
    content = re.sub(pattern, replace_message_with_data, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also handle error cases - replace message= with error= when success=False
    error_pattern = r'(success=False[^)]*?)data=\{"message":\s*([^}]+)\}'
    content = re.sub(error_pattern, r'\1error=\2', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")
        return True
    return False

def main():
    """Fix all nexus files"""
    
    base_dir = r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI"
    
    nexus_files = [
        "UIP/nexus/uip_nexus.py",
        "AGP/nexus/agp_nexus.py", 
        "LOGOS_Agent/nexus/agent_nexus.py",
        "GUI/nexus/gui_nexus.py"
    ]
    
    for nexus_file in nexus_files:
        file_path = os.path.join(base_dir, nexus_file)
        if os.path.exists(file_path):
            try:
                if fix_nexus_response_file(file_path):
                    print(f"✅ Successfully fixed {nexus_file}")
                else:
                    print(f"ℹ️ No changes needed for {nexus_file}")
            except Exception as e:
                print(f"❌ Error fixing {nexus_file}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    main()