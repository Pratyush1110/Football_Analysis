"""
report_generator.py
--------------------
Converts the stats dict produced by AnalyticsCollector into downloadable files:

  * JSON  – full stats snapshot
  * CSV   – player-level stats table
  * PDF   – formatted match report (requires reportlab)

All functions are safe to call even if optional dependencies are missing;
they return False / skip gracefully.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ── optional reportlab ────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ── colour palette ────────────────────────────────────────────────────────────
_DARK_BLUE = "#1a1a2e"
_MID_BLUE = "#16213e"
_LIGHT_ROW = "#eef2ff"

# ─────────────────────────────────────────────────────────────────────────────
# Plain-data exports
# ─────────────────────────────────────────────────────────────────────────────


def save_stats_json(stats: dict, path: str) -> None:
    """Serialise the full stats dict to pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


def save_stats_csv(stats: dict, path: str) -> None:
    """Write the player-level stats as a flat CSV file."""
    players = stats.get("players", {})
    if not players:
        return
    # Consistent column order
    fieldnames = [
        "player_id",
        "team",
        "total_distance_m",
        "avg_speed_kmh",
        "max_speed_kmh",
        "possession_frames",
        "possession_share_pct",
        "tracked_frames",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in sorted(players.values(), key=lambda x: x["player_id"]):
            writer.writerow(p)


# ─────────────────────────────────────────────────────────────────────────────
# PDF report
# ─────────────────────────────────────────────────────────────────────────────


def generate_pdf_report(stats: dict, path: str) -> bool:
    """
    Build a styled PDF report from ``stats``.

    Returns True on success, False when reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6,
        textColor=colors.HexColor(_DARK_BLUE),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor(_MID_BLUE),
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Football Match Analysis Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(_MID_BLUE)))
    story.append(Spacer(1, 0.4 * cm))

    # ── Match Summary ─────────────────────────────────────────────────────────
    match = stats.get("match", {})
    story.append(Paragraph("Match Summary", heading_style))
    match_data = [
        ["Metric", "Value"],
        ["Duration", match.get("duration_str", "N/A")],
        ["Total Frames", str(match.get("total_frames", "N/A"))],
        ["Players Detected", str(match.get("total_players_detected", "N/A"))],
        ["Team 1 Possession", f"{match.get('team1_possession_pct', 0.0):.1f}%"],
        ["Team 2 Possession", f"{match.get('team2_possession_pct', 0.0):.1f}%"],
        ["Overall Avg Speed", f"{match.get('overall_avg_speed_kmh', 0.0):.2f} km/h"],
    ]
    story.append(_build_table(match_data, col_widths=[8 * cm, 8 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    # ── Team Statistics ───────────────────────────────────────────────────────
    teams = stats.get("teams", {})
    story.append(Paragraph("Team Statistics", heading_style))
    team_data = [
        ["Team", "Players", "Possession %", "Total Distance (m)", "Avg Speed (km/h)"]
    ]
    for tid in sorted(teams.keys()):
        t = teams[tid]
        team_data.append(
            [
                f"Team {tid}",
                str(t.get("player_count", 0)),
                f"{t.get('possession_pct', 0.0):.1f}%",
                f"{t.get('total_distance_m', 0.0):.2f}",
                f"{t.get('avg_speed_kmh', 0.0):.2f}",
            ]
        )
    story.append(_build_table(team_data))
    story.append(Spacer(1, 0.4 * cm))

    # ── Player Statistics ─────────────────────────────────────────────────────
    players = stats.get("players", {})
    story.append(Paragraph("Player Statistics", heading_style))
    player_data = [
        [
            "ID",
            "Team",
            "Distance\n(m)",
            "Avg Speed\n(km/h)",
            "Max Speed\n(km/h)",
            "Possession\nFrames",
            "Tracked\nFrames",
        ]
    ]
    for pid in sorted(players.keys()):
        p = players[pid]
        player_data.append(
            [
                str(p.get("player_id", pid)),
                str(p.get("team", "?")),
                f"{p.get('total_distance_m', 0.0):.2f}",
                f"{p.get('avg_speed_kmh', 0.0):.2f}",
                f"{p.get('max_speed_kmh', 0.0):.2f}",
                str(p.get("possession_frames", 0)),
                str(p.get("tracked_frames", 0)),
            ]
        )
    col_w = [1.8 * cm, 1.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]
    story.append(_build_table(player_data, col_widths=col_w))

    doc.build(story)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_table(data: list, col_widths: Optional[list] = None) -> "Table":
    """Create a styled reportlab Table from a list-of-lists."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_MID_BLUE)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                # Data rows
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(_LIGHT_ROW)]),
                # Alignment & padding
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                # Grid
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c0c8e0")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(_MID_BLUE)),
            ]
        )
    )
    return tbl
