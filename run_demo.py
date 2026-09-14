#!/usr/bin/env python3
"""
Quick Launcher for Language Models and Agents (LMA) Web Interface
Run with: python run_demo.py
"""
import os
import sys
import webbrowser

if __name__ == "__main__":
    from interface.server import run_server
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            pass
    print(f"Opening browser at http://localhost:{port}...")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except:
        pass
    run_server(port)
