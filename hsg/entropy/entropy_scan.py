"""
Per-base BWT entropy scan around a genomic locus.

Computes H(next nucleotide | k-mer context) using an FM-index built over a
GRCh38 chromosome, then repeats the same analysis on an NLTK English corpus
for comparison.

Two scan modes
--------------
sliding-window
    Fix context length k.  Slide across every base in the query region and
    compute the Shannon entropy of the four nucleotide (or character)
    continuation probabilities at each position.

context-length
    Fix a set of representative positions.  For each, sweep k from 1 to
    max_k and record how entropy changes as context grows.

Usage
-----
Build the Rust extension first::

    cd bwt && maturin develop --release

Then run::

    python -m hsg.entropy.entropy_scan \\
        --chrom chr12 \\
        --center-start 65824760 \\
        --center-end   65824860 \\
        --flank        6000 \\
        --k            6 \\
        --max-k        12
"""

import math
import os
import sys
import argparse
import textwrap
from typing import Optional, Sequence

import nltk
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich import box

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _entropy(probs: dict[str, float]) -> float:
    """Shannon entropy in bits from a probability dict (0 log 0 = 0)."""
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _sliding_window_entropy(
    index,
    sequence: str,
    k: int,
    alphabet: tuple[str, ...],
    desc: str = "",
) -> list[float]:
    """For each position i in sequence[k:], compute H(next | seq[i-k:i]).

    Returns a list of length len(sequence)-k.  Positions < k have no context
    and are omitted.
    """
    n = len(sequence)
    entropies: list[float] = []

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{desc}[/cyan] sliding k={k}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("scan", total=n - k)
        for i in range(k, n):
            ctx = sequence[i - k : i]
            probs = index.next_token_probs(ctx, [ctx + a for a in alphabet])
            entropies.append(_entropy(probs))
            progress.advance(task)

    return entropies


def _context_length_entropy(
    index,
    sequence: str,
    positions: list[int],
    max_k: int,
    alphabet: tuple[str, ...],
) -> dict[int, list[float]]:
    """For each k in 1..max_k, compute entropy at each of `positions`.

    Returns {k: [entropy_at_pos_0, entropy_at_pos_1, ...]}.
    """
    results: dict[int, list[float]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]context-length scan[/cyan]"),
        BarColumn(bar_width=30),
        TextColumn("k={task.fields[k]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("ctx", total=max_k, k=1)
        for k in range(1, max_k + 1):
            progress.update(task, k=k)
            row: list[float] = []
            for pos in positions:
                if pos < k or pos >= len(sequence):
                    row.append(float("nan"))
                    continue
                ctx = sequence[pos - k : pos]
                probs = index.next_token_probs(ctx, [ctx + a for a in alphabet])
                row.append(_entropy(probs))
            results[k] = row
            progress.advance(task)
    return results


# ---------------------------------------------------------------------------
# ASCII sparkline renderer
# ---------------------------------------------------------------------------

_BLOCKS = " ▁▂▃▄▅▆▇█"
_SPARK_WIDTH = 80  # terminal columns for the sparkline


def _sparkline(values: list[float], width: int = _SPARK_WIDTH) -> str:
    """Downsample `values` to `width` characters and render as a block sparkline."""
    if not values:
        return ""
    # Pool into `width` buckets by averaging
    n = len(values)
    pooled: list[float] = []
    for i in range(width):
        lo = int(i * n / width)
        hi = int((i + 1) * n / width)
        hi = max(hi, lo + 1)
        chunk = [v for v in values[lo:hi] if not math.isnan(v)]
        pooled.append(sum(chunk) / len(chunk) if chunk else 0.0)

    vmin = min(pooled)
    vmax = max(pooled)
    span = vmax - vmin or 1.0
    chars = []
    for v in pooled:
        idx = int((v - vmin) / span * (len(_BLOCKS) - 1))
        chars.append(_BLOCKS[idx])
    return "".join(chars)


def _bar(value: float, vmax: float, width: int = 20) -> str:
    filled = int(value / vmax * width) if vmax > 0 else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_sliding_results(
    label: str,
    entropies: list[float],
    k: int,
    region_label: str,
) -> None:
    valid = [e for e in entropies if not math.isnan(e)]
    if not valid:
        console.print("[red]No valid entropy values.[/red]")
        return

    mean_h = sum(valid) / len(valid)
    min_h = min(valid)
    max_h = max(valid)

    console.print(Rule(f"[bold]{label}[/bold] — sliding window  k={k}"))
    console.print(f"Region : {region_label}")
    console.print(f"Bases  : {len(entropies)}  (context requires first {k} bases)")
    console.print(
        f"Entropy: mean={mean_h:.3f} bits  min={min_h:.3f}  max={max_h:.3f}"
    )
    console.print()
    spark = _sparkline(entropies)
    # Colour: low entropy = red (constrained), high = green (variable)
    console.print(f"[dim]low[/dim]  [red]{spark[:len(spark)//3]}[/red]"
                  f"[yellow]{spark[len(spark)//3:2*len(spark)//3]}[/yellow]"
                  f"[green]{spark[2*len(spark)//3:]}[/green]  [dim]high[/dim]")
    console.print(f"       [dim]{'◄ 5′':^{_SPARK_WIDTH // 2}}{'3′ ►':^{_SPARK_WIDTH // 2}}[/dim]")
    console.print()


def _print_context_results(
    label: str,
    ctx_results: dict[int, list[float]],
    position_labels: list[str],
) -> None:
    console.print(Rule(f"[bold]{label}[/bold] — context-length sweep"))

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("k", style="dim", width=4)
    for lbl in position_labels:
        table.add_column(lbl, justify="right", width=14)
    table.add_column("bar (mean)", width=24)

    ks = sorted(ctx_results)
    all_means = []
    for k in ks:
        vals = [v for v in ctx_results[k] if not math.isnan(v)]
        mean_v = sum(vals) / len(vals) if vals else 0.0
        all_means.append(mean_v)

    vmax = max(all_means) if all_means else 1.0

    for k, mean_v in zip(ks, all_means):
        cells = []
        for v in ctx_results[k]:
            cells.append(f"{v:.3f}" if not math.isnan(v) else "  —  ")
        bar = _bar(mean_v, vmax)
        table.add_row(str(k), *cells, f"{bar} {mean_v:.3f}")

    console.print(table)
    console.print()


# All standard GRCh38 chromosomes (index default when --idx-chroms not set).
_ALL_CHROMS: list = (
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)

# Default anchor locus.
_DEFAULT_ANCHOR_CHROM = "chr12"
_DEFAULT_ANCHOR_START = 65_824_760
_DEFAULT_ANCHOR_END   = 65_824_860


# ---------------------------------------------------------------------------
# Genomic analysis
# ---------------------------------------------------------------------------

def run_genomic(
    idx_chroms: list,
    anchor_chrom: str,
    anchor_start: int,
    anchor_end: int,
    flank: int,
    k: int,
    max_k: int,
    n_ctx_positions: int = 5,
) -> None:
    from hsg.entropy.bwt import build_genome_index, DNA_ALPHABET

    region_start = max(0, anchor_start - flank)
    region_end = anchor_end + flank
    region_label = f"{anchor_chrom}:{region_start:,}-{region_end:,}"

    console.print(Panel(
        textwrap.dedent(f"""\
            Indexed chroms : [bold]{', '.join(idx_chroms)}[/bold]
            Anchor locus   : {anchor_chrom}:{anchor_start:,}-{anchor_end:,}
            Flanks         : ±{flank:,} bp
            Scan window    : {region_label}  ({region_end - region_start:,} bp)
            Context k      : {k}   max_k : {max_k}"""),
        title="[bold green]Genomic entropy scan[/bold green]",
        border_style="green",
    ))

    # --- Build index over all requested chromosomes ---
    with console.status(f"[green]Building FM-index over {', '.join(idx_chroms)} …[/green]"):
        index = build_genome_index(idx_chroms)

    console.print(f"[green]✓[/green] Index built — {index}\n")

    # --- Fetch the analysis window from the anchor chromosome ---
    with console.status("[green]Fetching analysis window from SeqRepo…[/green]"):
        from biocommons.seqrepo import SeqRepo
        sr = SeqRepo(os.environ["SEQREPO_PATH"])
        sequence = str(sr[f"GRCh38:{anchor_chrom}"][region_start:region_end]).upper()

    console.print(f"[green]✓[/green] Sequence fetched: {len(sequence):,} bp\n")

    # --- Sliding window ---
    entropies = _sliding_window_entropy(
        index, sequence, k, DNA_ALPHABET, desc="genomic"
    )
    _print_sliding_results("Genomic", entropies, k, region_label)

    # --- Context-length sweep ---
    step = max(1, len(sequence) // (n_ctx_positions + 1))
    ctx_positions = [step * (i + 1) for i in range(n_ctx_positions)]
    pos_labels = [
        f"{anchor_chrom}:{region_start + p:,}" for p in ctx_positions
    ]

    ctx_results = _context_length_entropy(
        index, sequence, ctx_positions, max_k, DNA_ALPHABET
    )
    _print_context_results("Genomic", ctx_results, pos_labels)


# ---------------------------------------------------------------------------
# NLTK analysis
# ---------------------------------------------------------------------------

def _nltk_corpus_text(corpus_name: str, fileid: str) -> str:
    """Return raw character text from an NLTK corpus file."""
    corpus = getattr(nltk.corpus, corpus_name)
    return corpus.raw(fileid)


def run_nltk(
    corpus_name: str,
    fileid: str,
    window_chars: int,
    k: int,
    max_k: int,
    n_ctx_positions: int = 5,
) -> None:
    console.print(Panel(
        textwrap.dedent(f"""\
            Corpus  : [bold]{corpus_name}[/bold]
            File    : {fileid}
            Window  : first {window_chars:,} characters
            Context k : {k}   max_k : {max_k}"""),
        title="[bold blue]NLTK English corpus entropy scan[/bold blue]",
        border_style="blue",
    ))

    nltk.download(corpus_name, quiet=True)
    full_text = _nltk_corpus_text(corpus_name, fileid)

    # Use only the first `window_chars` characters (analogous to a genomic window).
    text = full_text[:window_chars].upper()
    # Derive alphabet from the window (printable chars only).
    alphabet = tuple(sorted(set(text) - {"\n", "\r", "\t"}))

    console.print(f"[blue]✓[/blue] Corpus loaded — {len(text):,} chars  |  "
                  f"alphabet size: {len(alphabet)}\n")

    with console.status("[blue]Building FM-index over English corpus…[/blue]"):
        from hsg.entropy.bwt import build_corpus_index
        index = build_corpus_index(text)

    console.print(f"[blue]✓[/blue] Index built — {index}\n")

    # --- Sliding window ---
    entropies = _sliding_window_entropy(
        index, text, k, alphabet, desc="english"
    )
    _print_sliding_results("English", entropies, k, f"{corpus_name}/{fileid}[:{window_chars}]")

    # --- Context-length sweep ---
    step = max(1, len(text) // (n_ctx_positions + 1))
    ctx_positions = [step * (i + 1) for i in range(n_ctx_positions)]
    pos_labels = [f"char {p:,}" for p in ctx_positions]

    ctx_results = _context_length_entropy(
        index, text, ctx_positions, max_k, alphabet
    )
    _print_context_results("English", ctx_results, pos_labels)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hsg.entropy.entropy_scan",
        description=textwrap.dedent("""\
            Per-base BWT entropy scan.

            Builds an FM-index over a GRCh38 chromosome and/or an NLTK
            English corpus, then runs two scan methods:
              1. Sliding-window  — fixed context k, scan every base.
              2. Context-length  — fixed positions, sweep k from 1 to max_k.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("genomic options")
    g.add_argument(
        "--idx-chroms", nargs="+", default=None, metavar="CHR",
        help="GRCh38 chromosome(s) to build the FM-index over. "
             "Pass one or more (e.g. --idx-chroms chr1 chr2 chrX). "
             "Omit to index the full genome (chr1-22, chrX, chrY).",
    )
    g.add_argument(
        "--anchor", default=None, metavar="REGION",
        help="Anchor locus for the scan window, as CHROM:START-END "
             "(0-based, e.g. chr12:65824760-65824860). "
             f"Default: {_DEFAULT_ANCHOR_CHROM}:{_DEFAULT_ANCHOR_START}-{_DEFAULT_ANCHOR_END}.",
    )
    g.add_argument("--flank", type=int, default=6_000, metavar="INT",
                   help="Bases upstream/downstream of anchor locus (default: 6000)")

    n = p.add_argument_group("NLTK options")
    n.add_argument("--nltk-corpus", default="gutenberg", metavar="NAME",
                   help="NLTK corpus name (default: gutenberg)")
    n.add_argument("--nltk-fileid", default="melville-moby_dick.txt", metavar="FILE",
                   help="File within the corpus (default: melville-moby_dick.txt)")
    n.add_argument("--nltk-window", type=int, default=200_000, metavar="INT",
                   help="Characters of English text to index (default: 200000)")

    s = p.add_argument_group("scan parameters")
    s.add_argument("--k", type=int, default=6, metavar="INT",
                   help="Fixed context length for sliding-window scan (default: 6)")
    s.add_argument("--max-k", type=int, default=12, metavar="INT",
                   help="Max context length for context-sweep scan (default: 12)")

    p.add_argument("--skip-genomic", action="store_true",
                   help="Skip the genomic analysis")
    p.add_argument("--skip-nltk", action="store_true",
                   help="Skip the NLTK analysis")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # --- Resolve idx-chroms ---
    idx_chroms = args.idx_chroms if args.idx_chroms else _ALL_CHROMS

    # --- Resolve anchor ---
    if args.anchor:
        try:
            chrom_part, coords = args.anchor.split(":")
            a_start, a_end = (int(x) for x in coords.split("-"))
        except ValueError:
            _build_parser().error(
                "--anchor must be CHROM:START-END, e.g. chr12:65824760-65824860"
            )
    else:
        chrom_part = _DEFAULT_ANCHOR_CHROM
        a_start    = _DEFAULT_ANCHOR_START
        a_end      = _DEFAULT_ANCHOR_END

    console.print()
    console.print(Rule("[bold white]BWT Entropy Scanner[/bold white]"))
    console.print()

    if not args.skip_genomic:
        run_genomic(
            idx_chroms,
            chrom_part,
            a_start,
            a_end,
            args.flank,
            args.k,
            args.max_k,
        )

    if not args.skip_nltk:
        run_nltk(
            args.nltk_corpus,
            args.nltk_fileid,
            args.nltk_window,
            args.k,
            args.max_k,
        )

    console.print(Rule("[dim]done[/dim]"))


if __name__ == "__main__":
    main()
