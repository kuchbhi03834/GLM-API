"""Entry point — run the OpenAI-compatible GLM server.

    python app.py            # serves on http://localhost:8001

Supports both:
  1. Free web session (chatglm.cn) via saved profile / interactive login.
  2. Zhipu AI BigModel Open Platform API via ZHIPU_API_KEY.
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8001"))
    print(f"Starting GLM OpenAI API on http://{host}:{port}")
    uvicorn.run(
        "server.api:app",
        host=host,
        port=port,
        reload=False,
    )
