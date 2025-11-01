#!/usr/bin/env python3
"""
Quick fix for remaining duplicate data parameters
"""

def fix_file_data_params(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'data={"message":' in line and i+1 < len(lines) and 'data=' in lines[i+1]:
            # Found duplicate data lines
            message_line = line
            data_line = lines[i+1]
            
            # Extract message part
            message_start = message_line.find('{"message":')
            message_end = message_line.find('},', message_start) + 1
            message_part = message_line[message_start:message_end]
            
            # Extract variable name from second line
            var_start = data_line.find('data=') + 5
            var_end = len(data_line.strip())
            var_name = data_line[var_start:var_end].strip()
            if var_name.endswith(','):
                var_name = var_name[:-1]
            
            # Create merged line
            indent = line[:line.find('data=')]
            new_line = f'{indent}data={{\n'
            fixed_lines.append(new_line)
            fixed_lines.append(f'{indent}    "message": {message_part[11:-1]},\n')  # Remove {"message": and }
            fixed_lines.append(f'{indent}    **{var_name}\n')
            fixed_lines.append(f'{indent}}}\n')
            
            i += 2  # Skip both lines
        else:
            fixed_lines.append(line)
            i += 1
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

# Fix LOGOS_Agent
fix_file_data_params(r"c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\LOGOS_AI\LOGOS_Agent\nexus\agent_nexus.py")
print("Fixed LOGOS_Agent nexus")