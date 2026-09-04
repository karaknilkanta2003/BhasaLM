"""
Interactive Command Line Interface (CLI) for Multilingual Transformer LMs
=========================================================================
Supports:
- Model H (Hindi ~25M parameters)
- Model L (Assamese ~25M parameters)
- Simultaneous multi-temperature generation comparison
- Dynamic temperature tuning, top-k, repetition penalty, and token limits
- Full UTF-8 support for Devanagari and Bengali-Assamese scripts
"""

import os
import sys
import io
import time
import json
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Reconfigure console streams for UTF-8 on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
import sentencepiece as spm

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common.model import TransformerLM, ModelConfig


# ==============================================================================
# ANSI Terminal Styling
# ==============================================================================
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_DARK = "\033[100m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET}"


# ==============================================================================
# Model Configurations & Registry
# ==============================================================================
LANGUAGE_CONFIGS = {
    "hindi": {
        "name": "Hindi (हिन्दी)",
        "code": "hi",
        "model_label": "Model H (High-Resource)",
        "config_path": os.path.join(SCRIPT_DIR, "results_hindi", "config.json"),
        "checkpoint_path": os.path.join(SCRIPT_DIR, "results_hindi", "checkpoints", "model_h_step_25000.pt"),
        "tokenizer_path": os.path.join(SCRIPT_DIR, "hindi", "tokenizer", "hindi_sp.model"),
        "sample_prompts": [
            "भारत का इतिहास",
            "विज्ञान और तकनीक",
            "प्राकृतिक सौंदर्य",
            "शिक्षा का महत्व",
            "हमारे देश में"
        ]
    },
    "assamese": {
        "name": "Assamese (অসমীয়া)",
        "code": "as",
        "model_label": "Model L (Low-Resource)",
        "config_path": os.path.join(SCRIPT_DIR, "results_assamese", "config.json"),
        "checkpoint_path": os.path.join(SCRIPT_DIR, "results_assamese", "checkpoints", "model_l_step_25000.pt"),
        "tokenizer_path": os.path.join(SCRIPT_DIR, "assamese", "tokenizer", "assamese_sp.model"),
        "sample_prompts": [
            "অসমৰ প্ৰাকৃতিক সৌন্দৰ্য",
            "ব্ৰহ্মপুত্ৰ নদীৰ",
            "অসমীয়া সাহিত্যৰ",
            "ইতিহাসৰ পাতত",
            "শিক্ষা আৰু সমাজ"
        ]
    }
}


def get_temperature_label(t: float) -> str:
    if t <= 1e-6:
        return "Greedy (Deterministic / Most Likely)"
    elif t <= 0.35:
        return "Focused (Low Entropy / Conservative)"
    elif t <= 0.75:
        return "Balanced (Coherent & Fluent)"
    elif t <= 1.1:
        return "Creative (Natural Diversity)"
    else:
        return "Exploratory (High Entropy / Experimental)"


# ==============================================================================
# Generation Engine
# ==============================================================================
class LMInferenceEngine:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.current_lang = None
        self.model = None
        self.config = None
        self.sp = None

    def load_language(self, lang_key: str):
        lang_key = lang_key.lower().strip()
        if lang_key in ["hi", "hindi", "h"]:
            target_key = "hindi"
        elif lang_key in ["as", "assamese", "asm", "l"]:
            target_key = "assamese"
        else:
            raise ValueError(f"Unknown language '{lang_key}'. Choose 'hindi' or 'assamese'.")

        if self.current_lang == target_key and self.model is not None:
            return  # Already loaded

        meta = LANGUAGE_CONFIGS[target_key]
        print(f"\n{colored('⚙ Loading', Style.BRIGHT_CYAN)} {meta['model_label']} [{meta['name']}]...")

        if not os.path.exists(meta["config_path"]):
            raise FileNotFoundError(f"Config file not found: {meta['config_path']}")
        if not os.path.exists(meta["checkpoint_path"]):
            raise FileNotFoundError(f"Checkpoint file not found: {meta['checkpoint_path']}")
        if not os.path.exists(meta["tokenizer_path"]):
            raise FileNotFoundError(f"Tokenizer file not found: {meta['tokenizer_path']}")

        t0 = time.time()
        # 1. Load Config
        with open(meta["config_path"], "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        self.config = ModelConfig.from_dict(cfg_dict)

        # 2. Build Model
        self.model = TransformerLM(self.config).to(self.device)

        # 3. Load Weights
        ckpt = torch.load(meta["checkpoint_path"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 4. Load SentencePiece Tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=meta["tokenizer_path"])

        self.current_lang = target_key
        elapsed = time.time() - t0
        print(f"{colored('✓ Loaded', Style.BRIGHT_GREEN)} {meta['name']} on {self.device.upper()} in {elapsed:.2f}s (~27.24M parameters).\n")

    @torch.no_grad()
    def generate_single(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 50,
        top_k: Optional[int] = 50,
        repetition_penalty: float = 1.15
    ) -> Dict[str, Any]:
        """
        Generates continuation for a prompt with specified temperature and settings.
        """
        if self.model is None or self.sp is None:
            raise RuntimeError("Model not initialized. Call load_language() first.")

        # Encode prompt
        input_ids = self.sp.encode_as_ids(prompt)
        if len(input_ids) == 0:
            input_ids = [self.sp.bos_id() if self.sp.bos_id() != -1 else 1]

        prompt_len = len(input_ids)
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        start_time = time.time()
        generated_token_ids = []

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _, _ = self.model(idx_cond)
            next_logits = logits[:, -1, :].clone()

            # Apply repetition penalty to previously generated tokens
            if repetition_penalty != 1.0:
                seen_tokens = set(idx[0].tolist())
                for tid in seen_tokens:
                    if next_logits[0, tid] > 0:
                        next_logits[0, tid] /= repetition_penalty
                    else:
                        next_logits[0, tid] *= repetition_penalty

            # Temperature scaling or greedy
            if temperature <= 1e-6:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_val = next_token.item()
            generated_token_ids.append(token_val)
            idx = torch.cat([idx, next_token], dim=1)

            # Check EOS
            if self.sp.eos_id() != -1 and token_val == self.sp.eos_id():
                break

        elapsed = time.time() - start_time
        full_tokens = idx[0].tolist()
        full_text = self.sp.decode(full_tokens)
        continuation_text = self.sp.decode(generated_token_ids)

        # Calculate metrics
        tok_count = len(generated_token_ids)
        tok_per_sec = (tok_count / elapsed) if elapsed > 0 else 0.0
        unique_tokens = len(set(generated_token_ids))
        diversity_ratio = (unique_tokens / tok_count) if tok_count > 0 else 1.0

        return {
            "temperature": temperature,
            "label": get_temperature_label(temperature),
            "prompt": prompt,
            "continuation": continuation_text,
            "full_text": full_text,
            "tokens_generated": tok_count,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tok_per_sec,
            "diversity_ratio": diversity_ratio
        }

    def generate_across_temperatures(
        self,
        prompt: str,
        temperatures: List[float],
        max_new_tokens: int = 50,
        top_k: Optional[int] = 50,
        repetition_penalty: float = 1.15
    ) -> List[Dict[str, Any]]:
        results = []
        for temp in temperatures:
            res = self.generate_single(
                prompt=prompt,
                temperature=temp,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            results.append(res)
        return results


# ==============================================================================
# Display & Formatting Helpers
# ==============================================================================
def print_banner():
    banner = f"""
{colored('╔══════════════════════════════════════════════════════════════════════════════╗', Style.BRIGHT_CYAN)}
{colored('║', Style.BRIGHT_CYAN)}  {colored('MULTILINGUAL TRANSFORMER LM — INTERACTIVE COMMAND LINE INTERFACE', Style.BOLD + Style.BRIGHT_WHITE)}   {colored('║', Style.BRIGHT_CYAN)}
{colored('║', Style.BRIGHT_CYAN)}  {colored('Model H (Hindi) & Model L (Assamese) | ~27.24M Parameters from Scratch', Style.DIM + Style.WHITE)}  {colored('║', Style.BRIGHT_CYAN)}
{colored('╚══════════════════════════════════════════════════════════════════════════════╝', Style.BRIGHT_CYAN)}
"""
    print(banner)


def print_help():
    help_text = f"""
{colored('Available Interactive Commands:', Style.BOLD + Style.BRIGHT_YELLOW)}
  {colored('/lang [hindi|assamese]', Style.BRIGHT_GREEN)}  - Switch between Hindi (Model H) and Assamese (Model L)
  {colored('/temps t1 t2 ...', Style.BRIGHT_GREEN)}        - Change the temperature list (e.g., /temps 0.0 0.5 1.0 1.5)
  {colored('/tokens [N]', Style.BRIGHT_GREEN)}             - Set max new tokens to generate (e.g., /tokens 40)
  {colored('/topk [K]', Style.BRIGHT_GREEN)}               - Set top-k sampling cutoff (e.g., /topk 50, 0 = off)
  {colored('/rep [val]', Style.BRIGHT_GREEN)}              - Set repetition penalty (e.g., /rep 1.15, 1.0 = off)
  {colored('/samples', Style.BRIGHT_GREEN)}                - Show example prompts in the current language
  {colored('/info', Style.BRIGHT_GREEN)}                   - Display current model architecture & active settings
  {colored('/help', Style.BRIGHT_GREEN)}                   - Show this help message
  {colored('exit', Style.BRIGHT_RED)} or {colored('quit', Style.BRIGHT_RED)}         - Exit the interface
"""
    print(help_text)


def display_comparison_results(prompt: str, results: List[Dict[str, Any]], lang_name: str):
    width = 80
    border = colored("─" * width, Style.BRIGHT_BLACK)
    
    print(f"\n{colored('═' * width, Style.BRIGHT_CYAN)}")
    print(f"{colored(' PROMPT (' + lang_name + '):', Style.BOLD + Style.BRIGHT_YELLOW)} {colored(prompt, Style.BOLD + Style.BRIGHT_WHITE)}")
    print(f"{colored('═' * width, Style.BRIGHT_CYAN)}")

    for i, res in enumerate(results):
        t = res["temperature"]
        label = res["label"]
        continuation = res["continuation"].strip()
        
        # Color-code based on temperature
        if t <= 1e-6:
            tag_color = Style.BRIGHT_BLUE
            bullet = "✦ [GREEDY]      "
        elif t <= 0.35:
            tag_color = Style.BRIGHT_CYAN
            bullet = "✦ [FOCUSED]     "
        elif t <= 0.75:
            tag_color = Style.BRIGHT_GREEN
            bullet = "✦ [BALANCED]    "
        elif t <= 1.1:
            tag_color = Style.BRIGHT_YELLOW
            bullet = "✦ [CREATIVE]    "
        else:
            tag_color = Style.BRIGHT_MAGENTA
            bullet = "✦ [EXPLORATORY] "

        header = f"{bullet} Temperature = {t:.2f}  |  {label}"
        print(f"\n{colored(header, tag_color + Style.BOLD)}")
        print(border)
        
        # Highlight prompt vs continuation
        display_text = f"{colored(prompt, Style.DIM + Style.WHITE)} {colored(continuation, Style.BOLD + Style.BRIGHT_WHITE)}"
        print(f"  {display_text}")
        print(border)
        
        # Diagnostics
        stats = (
            f"  {colored('Tokens:', Style.DIM)} {res['tokens_generated']} | "
            f"{colored('Time:', Style.DIM)} {res['elapsed_seconds']:.2f}s | "
            f"{colored('Speed:', Style.DIM)} {res['tokens_per_second']:.1f} tok/s | "
            f"{colored('Diversity:', Style.DIM)} {res['diversity_ratio']*100:.1f}% unique"
        )
        print(stats)

    print(f"\n{colored('═' * width, Style.BRIGHT_CYAN)}\n")


# ==============================================================================
# Interactive Shell Loop
# ==============================================================================
def run_interactive_cli(default_lang: str = "hindi", default_device: Optional[str] = None):
    print_banner()

    engine = LMInferenceEngine(device=default_device)

    # Language selection prompt if not specified
    selected_lang = default_lang
    try:
        engine.load_language(selected_lang)
    except Exception as e:
        print(f"{colored('Error loading default language:', Style.BRIGHT_RED)} {e}")
        # Fallback to hindi
        engine.load_language("hindi")
        selected_lang = "hindi"

    # Default generation parameters
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.4]
    max_tokens = 40
    top_k = 50
    repetition_penalty = 1.15

    meta = LANGUAGE_CONFIGS[engine.current_lang]
    print(f"{colored('Active Language:', Style.BOLD)} {meta['name']} ({meta['model_label']})")
    print(f"{colored('Active Temperatures:', Style.BOLD)} {temperatures}")
    print(f"{colored('Max New Tokens:', Style.BOLD)} {max_tokens} | {colored('Top-K:', Style.BOLD)} {top_k} | {colored('Repetition Penalty:', Style.BOLD)} {repetition_penalty}")
    print(f"{colored('Type your prompt or question below, or type /help for commands.', Style.DIM)}")
    print(f"{colored('Example prompts:', Style.DIM)} {', '.join(meta['sample_prompts'][:3])}\n")

    while True:
        try:
            prompt_indicator = f"{colored('[' + meta['name'] + ']', Style.BRIGHT_CYAN)} {colored('❯', Style.BRIGHT_YELLOW)} "
            user_input = input(prompt_indicator).strip()

            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{colored('Exiting Interactive LM CLI. Goodbye!', Style.BRIGHT_YELLOW)}")
                break

            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in ["/help", "/h", "/?"]:
                    print_help()
                elif cmd in ["/lang", "/switch", "/l"]:
                    if len(cmd_parts) > 1:
                        new_lang = cmd_parts[1].lower()
                        if new_lang in ["hindi", "hi", "h"]:
                            engine.load_language("hindi")
                        elif new_lang in ["assamese", "as", "asm", "l"]:
                            engine.load_language("assamese")
                        else:
                            print(f"{colored('Invalid language.', Style.BRIGHT_RED)} Use 'hindi' or 'assamese'.")
                    else:
                        # Toggle
                        target = "assamese" if engine.current_lang == "hindi" else "hindi"
                        engine.load_language(target)
                    meta = LANGUAGE_CONFIGS[engine.current_lang]
                elif cmd in ["/temps", "/temp", "/t"]:
                    if len(cmd_parts) > 1:
                        try:
                            new_temps = [float(x) for x in cmd_parts[1:]]
                            temperatures = sorted(list(set(new_temps)))
                            print(f"{colored('✓ Temperatures updated to:', Style.BRIGHT_GREEN)} {temperatures}")
                        except ValueError:
                            print(f"{colored('Invalid temperature format.', Style.BRIGHT_RED)} Provide numbers like: /temps 0.0 0.5 1.0")
                    else:
                        print(f"Current temperatures: {temperatures}")
                elif cmd in ["/tokens", "/len", "/maxlen"]:
                    if len(cmd_parts) > 1:
                        try:
                            max_tokens = max(1, min(200, int(cmd_parts[1])))
                            print(f"{colored('✓ Max new tokens updated to:', Style.BRIGHT_GREEN)} {max_tokens}")
                        except ValueError:
                            print(f"{colored('Invalid integer.', Style.BRIGHT_RED)} Example: /tokens 40")
                    else:
                        print(f"Current max new tokens: {max_tokens}")
                elif cmd in ["/topk", "/k"]:
                    if len(cmd_parts) > 1:
                        try:
                            val = int(cmd_parts[1])
                            top_k = None if val <= 0 else val
                            print(f"{colored('✓ Top-K updated to:', Style.BRIGHT_GREEN)} {top_k}")
                        except ValueError:
                            print(f"{colored('Invalid integer.', Style.BRIGHT_RED)} Example: /topk 50")
                    else:
                        print(f"Current top-k: {top_k}")
                elif cmd in ["/rep", "/reppenalty"]:
                    if len(cmd_parts) > 1:
                        try:
                            repetition_penalty = max(1.0, float(cmd_parts[1]))
                            print(f"{colored('✓ Repetition penalty updated to:', Style.BRIGHT_GREEN)} {repetition_penalty:.2f}")
                        except ValueError:
                            print(f"{colored('Invalid float.', Style.BRIGHT_RED)} Example: /rep 1.15")
                    else:
                        print(f"Current repetition penalty: {repetition_penalty}")
                elif cmd in ["/samples", "/sample"]:
                    print(f"\n{colored('Sample Prompts in ' + meta['name'] + ':', Style.BOLD + Style.BRIGHT_YELLOW)}")
                    for idx, sp_text in enumerate(meta["sample_prompts"], 1):
                        print(f"  [{idx}] {colored(sp_text, Style.BRIGHT_WHITE)}")
                    print()
                elif cmd in ["/info"]:
                    print(f"\n{colored('Active Model Configuration:', Style.BOLD + Style.BRIGHT_CYAN)}")
                    print(f"  • Language:            {meta['name']} ({meta['model_label']})")
                    print(f"  • Device:              {engine.device.upper()}")
                    print(f"  • Architecture:        Decoder-only Pre-LN Causal Transformer")
                    print(f"  • Total Parameters:    ~27.24M ({engine.config.vocab_size} vocab, {engine.config.d_model} d_model, {engine.config.n_layer} layers, {engine.config.n_head} heads)")
                    print(f"  • Active Temperatures: {temperatures}")
                    print(f"  • Max New Tokens:      {max_tokens}")
                    print(f"  • Top-K Threshold:     {top_k}")
                    print(f"  • Repetition Penalty:  {repetition_penalty}\n")
                elif cmd in ["/clear", "/cls"]:
                    os.system("cls" if os.name == "nt" else "clear")
                    print_banner()
                else:
                    print(f"{colored('Unknown command:', Style.BRIGHT_RED)} {cmd}. Type /help for available commands.")
                continue

            # Standard prompt generation across all temperatures
            print(f"{colored('Generating responses across temperatures...', Style.DIM)}", end="\r")
            results = engine.generate_across_temperatures(
                prompt=user_input,
                temperatures=temperatures,
                max_new_tokens=max_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            display_comparison_results(user_input, results, meta["name"])

        except KeyboardInterrupt:
            print(f"\n\n{colored('Interrupted. Exiting...', Style.BRIGHT_YELLOW)}")
            break
        except Exception as e:
            print(f"\n{colored('Error during generation:', Style.BRIGHT_RED)} {e}\n")


# ==============================================================================
# Main Entry Point & CLI Argument Parser
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Transformer LM Interactive CLI (Hindi & Assamese ~25M)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hindi",
        choices=["hindi", "hi", "assamese", "as"],
        help="Language model to load: 'hindi' (Model H) or 'assamese' (Model L)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-shot prompt to generate for non-interactive execution"
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.7, 1.0, 1.4],
        help="List of temperatures to evaluate (e.g. --temps 0.0 0.5 1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum new tokens to generate (default: 40)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K sampling threshold (default: 50, 0 to disable)"
    )
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty applied to logits (default: 1.15, 1.0 to disable)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device (default: auto-detect)"
    )

    args = parser.parse_args()

    # If single-shot prompt provided, run once and exit
    if args.prompt:
        engine = LMInferenceEngine(device=args.device)
        engine.load_language(args.lang)
        meta = LANGUAGE_CONFIGS[engine.current_lang]
        results = engine.generate_across_temperatures(
            prompt=args.prompt,
            temperatures=args.temps,
            max_new_tokens=args.max_tokens,
            top_k=None if args.top_k <= 0 else args.top_k,
            repetition_penalty=args.rep_penalty
        )
        display_comparison_results(args.prompt, results, meta["name"])
    else:
        # Launch full interactive shell
        run_interactive_cli(default_lang=args.lang, default_device=args.device)


if __name__ == "__main__":
    main()
