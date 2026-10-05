import os

from .app import create_app

if __name__ == "__main__":
    app = create_app()
    llm = app.config["STATE"]["llm"]
    host = os.getenv("SAATHI_HOST", "127.0.0.1")  # local only by default: notes never leave the machine
    port = int(os.getenv("SAATHI_PORT") or os.getenv("PORT") or 5000)
    print(f"Saathi is using: {llm.kind} / {llm.name}")
    if llm.kind == "cloud":
        print("Using a hosted model: note text is sent to that provider. Unset SAATHI_CLOUD_API_KEY to stay local.")
    if llm.kind == "mock":
        print("No Ollama server found: running in demo mode. See README to install a real model.")
    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
