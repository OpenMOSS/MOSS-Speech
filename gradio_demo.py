#!/usr/bin/env python3
"""
Moss Speech Demo - Multimodal Speech Interaction System 
Main Program Entry
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("./Matcha-TTS")

from utils.interface import MIMOInterface


def parse_args():
    parser = argparse.ArgumentParser(description="Moss Speech Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="fnlp/MOSS-Speech",
        help="the path of model",
    )
    parser.add_argument(
        '--codec_path',
        type=str,
        default='fnlp/MOSS-Speech-Codec',
        help="the path of codec",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="server address")
    parser.add_argument("--port", type=int, default=7860, help="server port")
    parser.add_argument("--share", action="store_true", help="weather reate a public link")
    return parser.parse_args()


def main():
    args = parse_args()

    # create demo
    interface = MIMOInterface(args.model_path)
    demo = interface.create_interface()

    print(f"🚀 running Moss Speech Demo...")
    print(f"📱 model path: {args.model_path}")
    print(f"🌐 server link: http://{args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=['./assets']
    )


if __name__ == "__main__":
    main()
