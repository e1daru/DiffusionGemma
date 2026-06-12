#!/usr/bin/env python3
import argparse
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "google/diffusiongemma-26b-a4b-it"

console = Console()


def build_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        console.print("[bold red]NVIDIA_API_KEY not set.[/] Copy .env.example → .env and add your key.")
        raise SystemExit(1)
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def run(prompt: str, max_tokens: int, temperature: float, stream: bool) -> None:
    client = build_client()
    messages = [{"role": "user", "content": prompt}]

    console.print(f"\n[dim]Model:[/] {MODEL}")
    console.print(f"[dim]Prompt:[/] {prompt}\n")

    start = time.monotonic()

    if stream:
        console.print("[bold cyan]Response (streaming):[/]")
        full = []
        with client.chat.completions.stream(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as s:
            for chunk in s:
                delta = chunk.choices[0].delta.content or ""
                console.print(delta, end="")
                full.append(delta)
        console.print()
        elapsed = time.monotonic() - start
        console.print(f"\n[dim]Done in {elapsed:.1f}s[/]")
    else:
        with console.status("Generating..."):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elapsed = time.monotonic() - start
        text = resp.choices[0].message.content
        console.print(Panel(text, title="Response", border_style="green"))
        usage = resp.usage
        console.print(
            f"[dim]Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out — {elapsed:.1f}s[/]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffusionGemma via NVIDIA API")
    parser.add_argument("--prompt", "-p", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they arrive")
    args = parser.parse_args()

    run(args.prompt, args.max_tokens, args.temperature, args.stream)


if __name__ == "__main__":
    main()
