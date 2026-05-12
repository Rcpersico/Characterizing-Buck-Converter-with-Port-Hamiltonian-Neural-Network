import xmlrpc.client
from pathlib import Path

class PLECSRPCSimulator:
    def __init__(self, model_path, port=1080):
        self.model_path = str(Path(model_path).resolve()).replace('\\', '/')
        self.model_name = Path(model_path).stem
        self.server = xmlrpc.client.ServerProxy(f'http://localhost:{port}/RPC2')
        
        try:
            self.server.plecs.load(self.model_path)
            print(f"✓ Loaded {self.model_name}")
        except xmlrpc.client.Fault as e:
            if "already" not in e.faultString.lower():
                raise

    def simulate(self, params, t_end):
        opt_struct = {
            'ModelVars': {k: float(v) for k, v in params.items()},
            'SolverOpts': {
                'StopTime': str(t_end),
            }
        }
        return self.server.plecs.simulate(self.model_name, opt_struct)


if __name__ == '__main__':
    sim = PLECSRPCSimulator(r'..\models\buck_converter.plecs')
    
    result = sim.simulate({
        'V_in_val':  12.0,
        'L_val':     100e-6,
        'C_val':     100e-6,
        'RL_val':    0.1,
        'ESR_val':   0.05,
        'Rload_val': 10.0,
        'D_val':     0.5,
    }, t_end=0.01)
    
    print("Result type:", type(result))
    print("Keys:", list(result.keys()) if isinstance(result, dict) else "not a dict")
    print("Preview:", str(result)[:500])