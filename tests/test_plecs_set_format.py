import xmlrpc.client

server = xmlrpc.client.ServerProxy('http://localhost:1080/RPC2')

model_path = 'C:/Users/rcper/OneDrive/Desktop/Research/Research2026/DCtoDC Converter Characterization with Deep Learning/Characterizing Buck Converter withPort Hamiltonian Neural Network/models/buck_converter.plecs'

# Make sure model is loaded
try:
    server.plecs.load(model_path)
    print("✓ Model loaded\n")
except Exception as e:
    print(f"Model load: {e}\n")

print("=== Testing plecs.set() argument formats ===\n")

# Test different argument combinations
test_cases = [
    # Format 1: (model, component, param, value)
    {
        'desc': '4 args: model, component, param, value',
        'args': (model_path, 'V_in', 'V', 12.0)
    },
    # Format 2: (model, component/param, value)
    {
        'desc': '3 args: model, "component/param", value',
        'args': (model_path, 'V_in/V', 12.0)
    },
    # Format 3: (model, [component, param], value)
    {
        'desc': '3 args: model, [component, param], value',
        'args': (model_path, ['V_in', 'V'], 12.0)
    },
    # Format 4: dict/struct
    {
        'desc': 'dict: {model, component, param, value}',
        'args': ({'model': model_path, 'component': 'V_in', 'parameter': 'V', 'value': 12.0},)
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['desc']}")
    print(f"  Args: {test['args']}")
    try:
        result = server.plecs.set(*test['args'])
        print(f"  ✓ SUCCESS! Result: {result}\n")
        print("="*60)
        print(f"FOUND WORKING FORMAT: {test['desc']}")
        print(f"Args: {test['args']}")
        print("="*60)
        break
    except xmlrpc.client.Fault as e:
        print(f"  ✗ Fault: {e.faultString}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("\n=== Testing plecs.simulate() formats ===\n")

sim_tests = [
    {
        'desc': '2 args: model, time',
        'args': (model_path, 0.001)
    },
    {
        'desc': '3 args: model, start_time, end_time',
        'args': (model_path, 0.0, 0.001)
    },
    {
        'desc': 'dict format',
        'args': ({'model': model_path, 'time': 0.001},)
    },
]

for i, test in enumerate(sim_tests, 1):
    print(f"Test {i}: {test['desc']}")
    print(f"  Args: {test['args']}")
    try:
        result = server.plecs.simulate(*test['args'])
        print(f"  ✓ SUCCESS! Result: {result}\n")
        break
    except xmlrpc.client.Fault as e:
        print(f"  ✗ Fault: {e.faultString}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("\n=== Getting model tree to see component structure ===\n")
try:
    tree = server.plecs.getModelTree(model_path)
    print("Model tree structure:")
    print(tree)
    print("\nThis shows the exact component paths to use!")
except Exception as e:
    print(f"Could not get model tree: {e}")