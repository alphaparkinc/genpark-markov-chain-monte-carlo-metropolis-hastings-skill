"""MCP stdio server for Metropolis-Hastings MCMC."""
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from client import MetropolisHastingsSampler

def handle_rpc(request):
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "sample_gaussian_mcmc",
                        "description": "Sample from target normal distribution N(target_mean, target_std) via Metropolis-Hastings",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "target_mean": {"type": "number", "default": 0.0},
                                "target_std": {"type": "number", "default": 1.0},
                                "num_samples": {"type": "integer", "default": 1000},
                                "seed": {"type": "integer", "default": 42}
                            }
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "sample_gaussian_mcmc":
            t_mean = float(args.get("target_mean", 0.0))
            t_std = float(args.get("target_std", 1.0))
            n_samples = int(args.get("num_samples", 1000))
            seed = int(args.get("seed", 42))
            log_p = lambda x: -0.5 * ((x - t_mean) / t_std)**2
            res = MetropolisHastingsSampler.sample_1d(log_p, x0=t_mean, proposal_std=t_std, num_samples=n_samples, seed=seed)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {name} not found"}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_rpc(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
