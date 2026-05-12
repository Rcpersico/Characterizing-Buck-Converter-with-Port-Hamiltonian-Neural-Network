import xmlrpc.client

server = xmlrpc.client.ServerProxy('http://localhost:1080/RPC2')

print("=== PLECS RPC Method Help ===\n")

methods = [
    'plecs.load',
    'plecs.get',
    'plecs.set',
    'plecs.simulate',
    'plecs.scope',
]

for method in methods:
    print(f"{method}:")
    try:
        help_text = server.system.methodHelp(method)
        print(f"  {help_text}\n")
    except Exception as e:
        print(f"  (no help available)\n")