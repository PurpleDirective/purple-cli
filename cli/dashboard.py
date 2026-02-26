"""
Purple Proving Ground -- Dashboard

Custom display for Purple CLI.
Design language: rounded panels, block-element bars, gemstone markers.
"""

import sys
from pathlib import Path

_cli_dir = str(Path(__file__).parent)
if _cli_dir not in sys.path:
    sys.path.insert(0, _cli_dir)

import tracker

# ── Colors ──────────────────────────────────────────────────────────────────
# Using 256-color for richer purple shades unavailable in basic ANSI 16
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ITALIC = "\033[3m"
C_PURPLE = "\033[38;5;141m"      # Soft lavender purple
C_DEEP_PURPLE = "\033[38;5;99m"  # Deeper purple for accents
C_CYAN = "\033[38;5;117m"        # Soft cyan
C_GREEN = "\033[38;5;114m"       # Soft green
C_YELLOW = "\033[38;5;221m"      # Warm amber
C_RED = "\033[38;5;167m"         # Soft coral red
C_WHITE = "\033[38;5;252m"       # Soft white
C_MUTED = "\033[38;5;245m"       # Muted gray
C_BORDER = "\033[38;5;60m"       # Subtle purple-gray for borders

# ── Box Drawing ─────────────────────────────────────────────────────────────
# Rounded corners for a modern, approachable feel
TL = "╭"  # top-left
TR = "╮"  # top-right
BL = "╰"  # bottom-left
BR = "╯"  # bottom-right
H  = "─"  # horizontal
V  = "│"  # vertical

# ── Markers ─────────────────────────────────────────────────────────────────
DIAMOND = "◆"      # Brand mark (gemstone)
DIAMOND_SM = "◇"   # Outline diamond
ARROW_R = "▸"      # Prompt arrow
DOT = "●"          # Status dot
DOT_O = "○"        # Empty dot


def _box_top(title: str = "", width: int = 52) -> str:
    """Render a rounded box top with optional title."""
    if title:
        title_str = f" {title} "
        line_len = width - 2 - len(title_str)
        return f"{C_BORDER}{TL}{H}{C_RESET}{C_PURPLE}{C_BOLD}{title_str}{C_RESET}{C_BORDER}{H * line_len}{TR}{C_RESET}"
    return f"{C_BORDER}{TL}{H * (width - 2)}{TR}{C_RESET}"


def _box_row(content: str, width: int = 52) -> str:
    """Render a row inside a box. Content can include ANSI codes."""
    # Calculate visible length (strip ANSI codes)
    import re
    visible = re.sub(r'\033\[[0-9;]*m', '', content)
    padding = width - 4 - len(visible)
    if padding < 0:
        padding = 0
    return f"{C_BORDER}{V}{C_RESET}  {content}{' ' * padding}{C_BORDER}{V}{C_RESET}"


def _box_empty(width: int = 52) -> str:
    """Empty row inside a box."""
    return f"{C_BORDER}{V}{' ' * (width - 2)}{V}{C_RESET}"


def _box_bottom(width: int = 52) -> str:
    """Render a rounded box bottom."""
    return f"{C_BORDER}{BL}{H * (width - 2)}{BR}{C_RESET}"


def _bar(value: float, width: int = 12) -> str:
    """Render a value (0.0-1.0) as a block-element bar."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    empty = width - filled

    if value >= 0.7:
        color = C_GREEN
    elif value >= 0.5:
        color = C_YELLOW
    else:
        color = C_RED

    if filled == 0 and empty > 0:
        return f"{C_MUTED}{'░' * empty}{C_RESET}"

    return f"{color}{'█' * filled}{C_RESET}{C_MUTED}{'░' * empty}{C_RESET}"


def _pct(value: float, good: float = 0.7, warn: float = 0.5) -> str:
    """Color a percentage based on thresholds."""
    text = f"{value*100:>3.0f}%"
    if value >= good:
        return f"{C_GREEN}{text}{C_RESET}"
    elif value >= warn:
        return f"{C_YELLOW}{text}{C_RESET}"
    return f"{C_RED}{text}{C_RESET}"


def _sovereignty_bar(score: int, width: int = 20) -> str:
    """Render the sovereignty score as a gradient bar with percentage."""
    filled = int(score / 100 * width)
    filled = max(0, min(width, filled))
    empty = width - filled

    # Gradient: deep purple for low, lavender for mid, green for high
    if score >= 70:
        color = C_GREEN
    elif score >= 40:
        color = C_PURPLE
    else:
        color = C_DEEP_PURPLE

    bar = f"{color}{'▓' * filled}{C_RESET}{C_MUTED}{'░' * empty}{C_RESET}"
    return f"{bar} {C_BOLD}{score}%{C_RESET}"


def render_compact(model: str, *, ollama_version: str = "",
                   tool_count: int = 0, server_count: int = 0) -> str:
    """Render the compact startup dashboard — one unified box."""
    tier = tracker.get_current_tier()
    tier_name = tracker.TIER_NAMES.get(tier, "Unknown")
    metrics = tracker.compute_metrics()
    sov = tracker.sovereignty_score(metrics)
    W = 54  # box width

    lines = [""]

    # ── Single unified box ──
    lines.append(_box_top(f"{DIAMOND} Purple CLI", W))

    # Model + Ollama version
    ver = f" {C_MUTED}· Ollama v{ollama_version}{C_RESET}" if ollama_version else ""
    lines.append(_box_row(f"{C_WHITE}{model}{C_RESET}{ver}", W))

    # Tools/servers summary
    tools_info = f"{tool_count} tools · {server_count} servers" if tool_count else "no tools"
    lines.append(_box_row(f"{C_MUTED}{tools_info}{C_RESET}", W))

    if metrics["window_size"] > 0:
        lines.append(_box_empty(W))

        # Tier + sovereignty bar
        lines.append(_box_row(
            f"{C_BOLD}{C_PURPLE}{DIAMOND}{C_RESET} {C_WHITE}{tier_name}{C_RESET}"
            f" {C_MUTED}({tier}){C_RESET}  {_sovereignty_bar(sov, width=14)}",
            W,
        ))

        lines.append(_box_empty(W))

        # Metrics — two columns
        lines.append(_box_row(
            f"{C_MUTED}TCR{C_RESET}  {_bar(metrics['tcr'])} {_pct(metrics['tcr'])}"
            f"   {C_MUTED}FTA{C_RESET}  {_bar(metrics['fta'])} {_pct(metrics['fta'])}",
            W,
        ))
        lines.append(_box_row(
            f"{C_MUTED}TCSR{C_RESET} {_bar(metrics['tcsr'])} {_pct(metrics['tcsr'])}"
            f"   {C_MUTED}UOR{C_RESET}  {_bar(metrics['uor'])} {_pct(metrics['uor'], good=0.0, warn=0.3)}",
            W,
        ))

        lines.append(_box_empty(W))

        # Task counts
        lines.append(_box_row(
            f"{C_MUTED}{metrics['total_tasks']} tasks{C_RESET}"
            f" {C_MUTED}{DIAMOND_SM}{C_RESET} "
            f"{C_MUTED}{metrics['overridden_tasks']} overrides{C_RESET}",
            W,
        ))

        # Promotion notice or next-tier info
        promo = tracker.check_promotion()
        if promo:
            promo_name = tracker.TIER_NAMES.get(promo, promo)
            lines.append(_box_empty(W))
            lines.append(_box_row(
                f"{C_GREEN}{C_BOLD}{DIAMOND} PROMOTION READY{C_RESET} {C_WHITE}{ARROW_R} {promo_name}{C_RESET} {C_MUTED}/promote{C_RESET}",
                W,
            ))
        else:
            thresholds = tracker.GRADUATION.get(tier)
            if thresholds:
                min_tasks, min_tcr, min_fta, max_uor = thresholds
                gaps = []
                if metrics["window_size"] < min_tasks:
                    gaps.append(f"{min_tasks - metrics['window_size']} more tasks")
                if metrics["tcr"] < min_tcr:
                    gaps.append(f"TCR {ARROW_R} {min_tcr*100:.0f}%")
                if min_fta > 0 and metrics["fta"] < min_fta:
                    gaps.append(f"FTA {ARROW_R} {min_fta*100:.0f}%")
                if gaps:
                    lines.append(_box_row(
                        f"{C_MUTED}Next: {' · '.join(gaps)}{C_RESET}",
                        W,
                    ))
    else:
        # No data yet — just show tier cleanly
        lines.append(_box_row(
            f"{C_PURPLE}{DIAMOND}{C_RESET} {C_WHITE}{tier_name}{C_RESET} {C_MUTED}({tier}){C_RESET}",
            W,
        ))

    lines.append(_box_empty(W))
    lines.append(_box_row(f"{C_MUTED}/help for commands{C_RESET}", W))
    lines.append(_box_bottom(W))

    return "\n".join(lines)


def render_full() -> str:
    """Render the full stats view for /stats command."""
    tier = tracker.get_current_tier()
    tier_name = tracker.TIER_NAMES.get(tier, "Unknown")
    metrics = tracker.compute_metrics(window=50)
    sov = tracker.sovereignty_score(metrics)
    history = tracker.get_tier_history(limit=5)
    W = 54

    lines = [""]

    # ── Full Stats Header ──
    lines.append(_box_top(f"{DIAMOND} Proving Ground", W))
    lines.append(_box_row(f"{C_BOLD}{C_WHITE}Full Statistics{C_RESET}", W))
    lines.append(_box_bottom(W))
    lines.append("")

    # ── Current State ──
    lines.append(f"  {C_BOLD}{C_PURPLE}{DIAMOND}{C_RESET} {C_WHITE}{tier_name}{C_RESET} {C_MUTED}({tier}){C_RESET}")
    lines.append(f"  Sovereignty  {_sovereignty_bar(sov)}")
    lines.append("")

    # ── Metrics Detail ──
    lines.append(f"  {C_BOLD}{C_WHITE}Metrics{C_RESET} {C_MUTED}(last {metrics['window_size']} tasks){C_RESET}")
    lines.append(f"  {C_BORDER}{H * 40}{C_RESET}")
    lines.append(f"  {C_MUTED}Task Completion{C_RESET}    {_bar(metrics['tcr'], 16)} {_pct(metrics['tcr'])}")
    lines.append(f"  {C_MUTED}First-Try Accuracy{C_RESET} {_bar(metrics['fta'], 16)} {_pct(metrics['fta'])}")
    lines.append(f"  {C_MUTED}Tool Call Success{C_RESET}  {_bar(metrics['tcsr'], 16)} {_pct(metrics['tcsr'])}")
    lines.append(f"  {C_MUTED}User Overrides{C_RESET}     {_bar(metrics['uor'], 16)} {_pct(metrics['uor'], good=0.0, warn=0.3)}")
    if metrics["avg_rating"] > 0:
        stars = int(metrics["avg_rating"])
        lines.append(f"  {C_MUTED}Avg Rating{C_RESET}         {C_YELLOW}{'★' * stars}{'☆' * (5 - stars)}{C_RESET} {metrics['avg_rating']:.1f}")
    lines.append(f"  {C_MUTED}Total Tasks{C_RESET}        {C_WHITE}{metrics['total_tasks']}{C_RESET}")
    lines.append("")

    # ── Tier History ──
    if history:
        lines.append(f"  {C_BOLD}{C_WHITE}Tier History{C_RESET}")
        lines.append(f"  {C_BORDER}{H * 40}{C_RESET}")
        for h in history:
            date = h['changed_at'][:10]
            direction = f"{C_RED}▼{C_RESET}" if tracker.TIERS.index(h.get('new_tier', 'T0')) < tracker.TIERS.index(h.get('old_tier', 'T0')) else f"{C_GREEN}▲{C_RESET}"
            lines.append(
                f"  {C_MUTED}{date}{C_RESET}  {direction} {h['old_tier']} {ARROW_R} {h['new_tier']}  {C_MUTED}{h['reason'][:30]}{C_RESET}"
            )
        lines.append("")

    # ── Graduation Requirements ──
    lines.append(f"  {C_BOLD}{C_WHITE}Next Tier Requirements{C_RESET}")
    lines.append(f"  {C_BORDER}{H * 40}{C_RESET}")
    thresholds = tracker.GRADUATION.get(tier)
    if thresholds:
        min_tasks, min_tcr, min_fta, max_uor = thresholds

        def _check(val, target, higher_better=True):
            passed = val >= target if higher_better else val <= target
            return f"{C_GREEN}{DOT}{C_RESET}" if passed else f"{C_RED}{DOT_O}{C_RESET}"

        lines.append(f"  {_check(metrics['window_size'], min_tasks)} Tasks    {metrics['window_size']}/{min_tasks}")
        lines.append(f"  {_check(metrics['tcr'], min_tcr)} TCR      {metrics['tcr']*100:.0f}%/{min_tcr*100:.0f}%")
        if min_fta > 0:
            lines.append(f"  {_check(metrics['fta'], min_fta)} FTA      {metrics['fta']*100:.0f}%/{min_fta*100:.0f}%")
        lines.append(f"  {_check(metrics['uor'], max_uor, False)} UOR      {metrics['uor']*100:.0f}%/{max_uor*100:.0f}% max")
    elif tier == "T4":
        lines.append(f"  {C_GREEN}{DIAMOND} Sovereign tier reached.{C_RESET}")

    lines.append("")
    return "\n".join(lines)
