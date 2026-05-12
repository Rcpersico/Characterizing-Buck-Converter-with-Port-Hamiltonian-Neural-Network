import xmlrpc.client
import json

server = xmlrpc.client.ServerProxy('http://localhost:1080/RPC2')

model_path = 'C:/Users/rcper/OneDrive/Desktop/Research/Research2026/DCtoDC Converter Characterization with Deep Learning/Characterizing Buck Converter withPort Hamiltonian Neural Network/models/buck_converter.plecs'

print("=== Getting Model Tree ===\n")

tree_str = server.plecs.getModelTree(model_path)
tree = json.loads(tree_str)

def find_components(node, path="", components=[]):
    """Recursively find all components with their paths."""
    if isinstance(node, dict):
        # Check if this is a component
        if 'name' in node and 'type' in node:
            comp_type = node.get('type', '')
            comp_name = node.get('name', '')
            
            # Skip properties
            if node.get('kind') != 'property':
                full_path = f"{path}/{comp_name}" if path else comp_name
                components.append({
                    'path': full_path,
                    'name': comp_name,
                    'type': comp_type
                })
        
        # Recurse into children
        if 'children' in node:
            for child in node['children']:
                new_path = f"{path}/{node.get('name', '')}" if node.get('kind') != 'property' else path
                find_components(child, new_path, components)
    
    return components

components = find_components(tree)

print(f"Found {len(components)} components:\n")
for comp in components[:50]:  # Show first 50
    print(f"  Path: {comp['path']}")
    print(f"  Type: {comp['type']}")
    print()

print("\n=== Looking for our components ===\n")

targets = ['V_in', 'L', 'C', 'R2', 'R3', 'Load', 'Pulse Generator']
for target in targets:
    matches = [c for c in components if target in c['name']]
    if matches:
        print(f"{target}:")
        for m in matches:
            print(f"  Path: {m['path']}")
            print(f"  Type: {m['type']}")
    else:
        print(f"{target}: NOT FOUND")
    print()