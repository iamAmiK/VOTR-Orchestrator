"""python -m api"""
import uvicorn
from api.app import create_app

if __name__ == "__main__":
    from orchestrator.config import load_config
    cfg = load_config()
    uvicorn.run(create_app(cfg), host=cfg.api_host, port=cfg.api_port, reload=False)
