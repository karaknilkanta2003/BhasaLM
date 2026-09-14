"""
Interactive Demo Server for Hindi Model H & Assamese Model L
============================================================
Lightweight zero-dependency HTTP server that hosts the web interface and serves
causal language model inference directly from local checkpoints.
"""

import os
import sys
import json
import time
import math
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Dict, Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn.functional as F
import sentencepiece as spm

# Add workspace to sys.path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from common.model import TransformerLM, ModelConfig

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Manager Singleton (Lazy loading & In-Memory Caching)
class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache: Dict[str, Any] = {}
        print(f"[ModelManager] Hardware Accelerator: {self.device.upper()}")

    def get_model(self, lang: str, model_type: str):
        key = f"{lang}_{model_type}"
        if key in self.cache:
            return self.cache[key]

        print(f"[ModelManager] Loading {lang.upper()} ({model_type.upper()}) into memory...")
        start = time.time()

        # Resolve paths
        if lang == "hindi":
            sp_path = os.path.join(WORKSPACE_DIR, "hindi", "tokenizer", "hindi_sp.model")
            if model_type == "finetuned":
                ckpt_path = os.path.join(WORKSPACE_DIR, "results_hindi_reasoning", "checkpoints", "model_h_finetuned_best.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(WORKSPACE_DIR, "results_hindi_reasoning", "checkpoints", "model_h_finetuned.pt")
            else:
                ckpt_path = os.path.join(WORKSPACE_DIR, "results_hindi", "checkpoints", "model_h_step_25000.pt")
        else:
            sp_path = os.path.join(WORKSPACE_DIR, "assamese", "tokenizer", "assamese_sp.model")
            if model_type == "finetuned":
                ckpt_path = os.path.join(WORKSPACE_DIR, "results_assamese_reasoning", "checkpoints", "model_l_finetuned_best.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(WORKSPACE_DIR, "results_assamese_reasoning", "checkpoints", "model_l_finetuned.pt")
            else:
                ckpt_path = os.path.join(WORKSPACE_DIR, "results_assamese", "checkpoints", "model_l_step_25000.pt")

        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"Tokenizer not found: {sp_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

        sp = spm.SentencePieceProcessor(model_file=sp_path)
        config = ModelConfig(vocab_size=len(sp), d_model=512, n_layer=6, n_head=8, d_ff=2048, max_seq_len=256)
        model = TransformerLM(config).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        raw_sd = ckpt.get("model_state_dict", ckpt)
        clean_sd = {}
        for k, v in raw_sd.items():
            clean_sd[k.replace("embed.", "embedding.")] = v
        model.load_state_dict(clean_sd, strict=False)
        model.eval()

        print(f"[ModelManager] Loaded {key} in {time.time() - start:.2f}s from {os.path.basename(ckpt_path)}")
        entry = {"model": model, "sp": sp, "config": config, "ckpt_path": os.path.basename(ckpt_path)}
        self.cache[key] = entry
        return entry

    @torch.no_grad()
    def generate(
        self,
        lang: str,
        model_type: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 12
    ) -> Dict[str, Any]:
        entry = self.get_model(lang, model_type)
        model = entry["model"]
        sp = entry["sp"]

        start_time = time.time()
        prompt_ids = sp.encode(prompt, out_type=int)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_token_ids = []
        curr = x

        for _ in range(max_tokens):
            idx_cond = curr if curr.size(1) <= model.config.max_seq_len else curr[:, -model.config.max_seq_len:]
            logits, _, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 1e-5:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            tok_id = next_tok.item()
            if tok_id == sp.eos_id():
                break

            generated_token_ids.append(tok_id)
            curr = torch.cat((curr, next_tok), dim=1)

        latency_ms = round((time.time() - start_time) * 1000, 1)
        completion_text = sp.decode(generated_token_ids)
        token_pieces = [sp.id_to_piece(tid) for tid in generated_token_ids]

        # Extract primary answer entity/token
        cleaned_words = completion_text.strip().split()
        first_word = cleaned_words[0] if cleaned_words else ""

        return {
            "prompt": prompt,
            "completion": completion_text,
            "full_text": prompt + completion_text,
            "first_word": first_word,
            "token_ids": generated_token_ids,
            "token_pieces": token_pieces,
            "latency_ms": latency_ms,
            "num_tokens": len(generated_token_ids),
            "model_ckpt": entry["ckpt_path"],
            "device": self.device
        }

MANAGER = ModelManager()

class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = {
                "device": MANAGER.device,
                "loaded_models": list(MANAGER.cache.keys()),
                "status": "ready"
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # Serve static files
        if path == "/" or path == "":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            file_path = os.path.join(STATIC_DIR, path.lstrip("/"))

        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "text/plain")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(body)
                lang = payload.get("lang", "hindi").lower()
                model_type = payload.get("model_type", "finetuned").lower()
                prompt = payload.get("prompt", "").strip()
                temperature = float(payload.get("temperature", 0.0))
                max_tokens = int(payload.get("max_tokens", 15))

                if not prompt:
                    raise ValueError("Prompt cannot be empty")

                result = MANAGER.generate(lang, model_type, prompt, temperature, max_tokens)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print("=" * 70)
    print(f"[SERVER] Indic Transformer LM Interface Running At: http://localhost:{port}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            pass
    run_server(port)
