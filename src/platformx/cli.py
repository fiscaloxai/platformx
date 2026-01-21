"""PlatformX Command Line Interface.

Usage:
    platformx <command> [options]

Commands:
    index       Index documents for retrieval
    query       Execute a RAG query
    raft        Generate RAFT training samples
    finetune    Fine-tune a model
    generate    Generate text with a model
    info        Show PlatformX information
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List

def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def cmd_index(args: argparse.Namespace) -> int:
    """Handle the 'index' command."""
    try:
        from platformx.api import index_documents
        result = index_documents(
            source=args.source,
            dataset_id=args.dataset_id,
            index_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_backend=args.embedding,
        )
        print(f"Indexed dataset '{args.dataset_id}' with {result['chunk_count']} chunks.")
        print(f"Index saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"Error indexing documents: {e}", file=sys.stderr)
        return 1

def cmd_query(args: argparse.Namespace) -> int:
    """Handle the 'query' command."""
    try:
        from platformx.api import rag_query
        results = rag_query(
            query=args.query,
            index_path=args.index,
            top_k=args.top_k,
            safety_check=not args.no_safety,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Top {args.top_k} results for query: '{args.query}'")
            for i, r in enumerate(results, 1):
                print(f"[{i}] Score: {r['score']:.3f} | {r['text']}")
        return 0
    except Exception as e:
        print(f"Error executing query: {e}", file=sys.stderr)
        return 1

def cmd_raft(args: argparse.Namespace) -> int:
    """Handle the 'raft' command."""
    try:
        from platformx.api import generate_raft_samples
        samples = generate_raft_samples(
            dataset_ids=args.datasets,
            index_path=args.index,
            output_path=args.output,
            samples_per_dataset=args.samples,
            positive_fraction=args.positive_fraction,
            seed=args.seed,
        )
        print(f"Generated {len(samples)} RAFT samples. Output: {args.output}")
        if args.json:
            print(json.dumps(samples, indent=2))
        return 0
    except Exception as e:
        print(f"Error generating RAFT samples: {e}", file=sys.stderr)
        return 1

def cmd_finetune(args: argparse.Namespace) -> int:
    """Handle the 'finetune' command."""
    try:
        from platformx.api import finetune
        report = finetune(
            base_model=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            batch_size=args.batch_size,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print(f"Finetune complete. Adapter ID: {report.get('adapter_id', 'N/A')}")
        print(f"Output directory: {args.output}")
        if not args.dry_run and 'metrics' in report:
            print(f"Metrics: {json.dumps(report['metrics'], indent=2)}")
        return 0
    except Exception as e:
        print(f"Error during finetuning: {e}", file=sys.stderr)
        return 1

def cmd_generate(args: argparse.Namespace) -> int:
    """Handle the 'generate' command."""
    try:
        from platformx.api import generate
        if args.prompt == "-":
            prompt = sys.stdin.read()
        else:
            prompt = args.prompt
        response = generate(
            prompt=prompt,
            model=args.model,
            adapter_path=args.adapter,
            backend=args.backend,
            provider=args.provider,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            safety_check=not args.no_safety,
        )
        if args.json:
            print(json.dumps(response, indent=2))
        else:
            print(f"Response: {response.get('text', response)}")
        return 0
    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        return 1

def cmd_info(args: argparse.Namespace) -> int:
    """Handle the 'info' command."""
    import platformx
    info = platformx.info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"PlatformX version: {info.get('version', 'N/A')}")
        print(f"Python version: {info.get('python_version', 'N/A')}")
        print(f"Module status: {info.get('modules', 'N/A')}")
    return 0

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="platformx",
        description="PlatformX - Production-quality LLM fine-tuning, RAG, and RAFT library"
    )
    parser.add_argument("--version", action="version", version="%(prog)s v0.1.0", help="Show PlatformX version and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # index
    p_index = subparsers.add_parser("index", help="Index documents for retrieval.")
    p_index.add_argument("source", type=str, help="Path to file or directory to index.")
    p_index.add_argument("-d", "--dataset-id", required=True, type=str, help="Dataset identifier.")
    p_index.add_argument("-o", "--output", type=str, default="./index.json", help="Output index path.")
    p_index.add_argument("--chunk-size", type=int, default=200, help="Words per chunk (default: 200).")
    p_index.add_argument("--chunk-overlap", type=int, default=50, help="Overlap words (default: 50).")
    p_index.add_argument("-e", "--embedding", type=str, default="tfidf", choices=["tfidf", "sentence-transformer"], help="Embedding backend.")
    p_index.set_defaults(func=cmd_index)

    # query
    p_query = subparsers.add_parser("query", help="Execute a RAG query.")
    p_query.add_argument("query", type=str, help="Query text.")
    p_query.add_argument("-i", "--index", required=True, type=str, help="Path to index.")
    p_query.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    p_query.add_argument("--no-safety", action="store_true", help="Disable safety checks.")
    p_query.add_argument("--json", action="store_true", help="Output as JSON.")
    p_query.set_defaults(func=cmd_query)

    # raft
    p_raft = subparsers.add_parser("raft", help="Generate RAFT training samples.")
    p_raft.add_argument("-d", "--datasets", required=True, nargs="+", type=str, help="Dataset IDs.")
    p_raft.add_argument("-i", "--index", required=True, type=str, help="Path to index.")
    p_raft.add_argument("-o", "--output", type=str, default="./raft_samples.json", help="Output file path.")
    p_raft.add_argument("-n", "--samples", type=int, default=10, help="Samples per dataset (default: 10).")
    p_raft.add_argument("--positive-fraction", type=float, default=0.6, help="Fraction of positive samples (default: 0.6).")
    p_raft.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_raft.add_argument("--json", action="store_true", help="Output as JSON.")
    p_raft.set_defaults(func=cmd_raft)

    # finetune
    p_finetune = subparsers.add_parser("finetune", help="Fine-tune a model.")
    p_finetune.add_argument("-m", "--model", required=True, type=str, help="Base model path or HuggingFace ID.")
    p_finetune.add_argument("-d", "--dataset", required=True, type=str, help="Training dataset path.")
    p_finetune.add_argument("-o", "--output", type=str, default="./output", help="Output directory.")
    p_finetune.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3).")
    p_finetune.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4).")
    p_finetune.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16).")
    p_finetune.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    p_finetune.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_finetune.add_argument("--dry-run", action="store_true", help="Validate configuration without training.")
    p_finetune.set_defaults(func=cmd_finetune)

    # generate
    p_generate = subparsers.add_parser("generate", help="Generate text with a model.")
    p_generate.add_argument("prompt", type=str, help="Prompt text (or '-' for stdin).")
    p_generate.add_argument("-m", "--model", type=str, help="Model path or name.")
    p_generate.add_argument("-a", "--adapter", type=str, help="Adapter path.")
    p_generate.add_argument("-b", "--backend", type=str, default="local", choices=["local", "api"], help="Backend type.")
    p_generate.add_argument("-p", "--provider", type=str, choices=["openai", "anthropic"], help="API provider.")
    p_generate.add_argument("--max-tokens", type=int, default=256, help="Max tokens (default: 256).")
    p_generate.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7).")
    p_generate.add_argument("--no-safety", action="store_true", help="Disable safety checks.")
    p_generate.add_argument("--json", action="store_true", help="Output as JSON.")
    p_generate.set_defaults(func=cmd_generate)

    # info
    p_info = subparsers.add_parser("info", help="Show PlatformX information.")
    p_info.add_argument("--json", action="store_true", help="Output as JSON.")
    p_info.set_defaults(func=cmd_info)

    return parser

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

__all__ = ["main", "create_parser"]
