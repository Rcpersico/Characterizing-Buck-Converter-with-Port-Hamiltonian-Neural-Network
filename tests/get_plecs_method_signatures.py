import xmlrpc.client

server = xmlrpc.client.ServerProxy('http://localhost:1080/RPC2')

print("=== PLECS RPC Method Signatures ===\n")

methods = [
    'plecs.load',
    'plecs.set',
    'plecs.get',
    'plecs.simulate',
    'plecs.scope',
]

for method in methods:
    print(f"\n{method}:")
    print("-" * 60)
    
    # Get help text
    try:
        help_text = server.system.methodHelp(method)
        print(help_text)
    except:
        print("(no help available)")
    
    # Try to call with wrong args to see error message
    print("\nTesting with dummy arguments to see expected format:")
    try:
        if method == 'plecs.load':
            server.plecs.load('test.plecs')
        elif method == 'plecs.set':
            server.plecs.set('model', 'path', 'value')
        elif method == 'plecs.get':
            server.plecs.get('model', 'path')
        elif method == 'plecs.simulate':
            server.plecs.simulate('model', 1.0)
        elif method == 'plecs.scope':
            server.plecs.scope('model', 'scope')
    except xmlrpc.client.Fault as e:
        # Error message often reveals expected format
        print(f"Fault: {e.faultString}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
print("\nAlso try calling plecs.getModelTree() to see model structure:")
try:
    tree = server.plecs.getModelTree(
        'C:\\Users\\rcper\\OneDrive\\Desktop\\Research\\Research2026\\DCtoDC Converter Characterization with Deep Learning\\Characterizing Buck Converter withPort Hamiltonian Neural Network\\models\\buck_converter.plecs'
    )
    print("\nModel tree:")
    print(tree)
except Exception as e:
    print(f"Error: {e}")