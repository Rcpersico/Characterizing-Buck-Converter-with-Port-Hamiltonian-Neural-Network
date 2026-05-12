import xmlrpc.client

server = xmlrpc.client.ServerProxy('http://localhost:1080/RPC2')

full_path = 'C:/Users/rcper/OneDrive/Desktop/Research/Research2026/DCtoDC Converter Characterization with Deep Learning/Characterizing Buck Converter withPort Hamiltonian Neural Network/models/buck_converter.plecs'

print("=== Testing Model Identifiers ===\n")

# Load model
print("Loading model...")
try:
    server.plecs.load(full_path)
    print("✓ Model loaded\n")
except Exception as e:
    print(f"Load error: {e}\n")

# Try different model identifiers
identifiers = [
    ('Full path', full_path),
    ('Filename only', 'buck_converter.plecs'),
    ('Name without extension', 'buck_converter'),
]

for desc, model_id in identifiers:
    print(f"Testing model identifier: {desc}")
    print(f"  Value: '{model_id}'")
    
    # Try to simulate with this identifier
    try:
        server.plecs.simulate(model_id, 0.001)
        print(f"  ✓ plecs.simulate() WORKS with '{model_id}'!")
        
        # Now try set with this identifier
        print(f"\n  Testing plecs.set() with this identifier...")
        try:
            server.plecs.set(model_id, 'V_in', 'V', 12.0)
            print(f"    ✓ plecs.set() also WORKS!")
            print(f"\n{'='*60}")
            print(f"SOLUTION FOUND!")
            print(f"  Model identifier: '{model_id}'")
            print(f"  Format: plecs.set('{model_id}', 'Component', 'Param', value)")
            print(f"{'='*60}")
            break
        except Exception as e:
            print(f"    ✗ plecs.set: {e}")
        
        break
        
    except xmlrpc.client.Fault as e:
        print(f"  ✗ Fault: {e.faultString}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("\n=== Also check PLECS documentation ===")
print("Look for RPC examples in:")
print("  C:\\Users\\rcper\\OneDrive\\Documents\\Plexim\\PLECS 5.0 (64 bit)\\Examples\\")
print("  C:\\Users\\rcper\\OneDrive\\Documents\\Plexim\\PLECS 5.0 (64 bit)\\Documentation\\")