"""
analytics_collector.py
-----------------------
Harvests per-player, per-team and match-level statistics from the fully-enriched
``tracks`` dict that the main pipeline produces.

Usage
-----
    from analytics_collector import AnalyticsCollector

    collector = AnalyticsCollector(fps=24.0)
    stats = collector.process_tracks(tracks, team_ball_control)
    # stats keys: "match", "players", "teams"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Any

import numpy as np


def _fmt_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class AnalyticsCollector:
    """
    Collect and aggregate player / team statistics from processed tracks.

    Parameters
    ----------
    fps : float
        Frames-per-second of the source video (default 24).
    """

    def __init__(self, fps: float = 24.0) -> None:
        self.fps = fps

    # ------------------------------------------------------------------
    def process_tracks(
        self,
        tracks: dict,
        team_ball_control: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Build analytics from the fully-enriched ``tracks`` dict and the
        ``team_ball_control`` array produced by the main pipeline.

        Returns
        -------
        dict with keys "match", "players", "teams".
        """
        # Accumulator: player_id → rolling data
        accum = defaultdict(
            lambda: {
                "team": None,
                "tracked_frames": 0,
                "possession_frames": 0,
                "speeds": [],           # km/h readings per frame
                "max_distance": 0.0,    # cumulative distance (monotone, take max)
            }
        )

        total_frames = len(tracks.get("players", []))

        for frame_tracks in tracks.get("players", []):
            for pid, info in frame_tracks.items():
                p = accum[pid]

                # Team assignment (first frame where it appears)
                if p["team"] is None and "team" in info:
                    p["team"] = info["team"]

                p["tracked_frames"] += 1

                # Ball possession flag
                if info.get("has_ball", False):
                    p["possession_frames"] += 1

                # Speed sample
                speed = info.get("speed")
                if speed is not None and speed > 0:
                    p["speeds"].append(float(speed))

                # Cumulative distance (pipeline stores monotone total per player)
                dist = info.get("distance")
                if dist is not None:
                    p["max_distance"] = max(p["max_distance"], float(dist))

        # ------------------------------------------------------------------
        # Player-level stats
        # ------------------------------------------------------------------
        players_stats: Dict[int, Dict[str, Any]] = {}
        for pid, p in accum.items():
            speeds = p["speeds"]
            players_stats[int(pid)] = {
                "player_id": int(pid),
                "team": int(p["team"]) if p["team"] is not None else None,
                "tracked_frames": p["tracked_frames"],
                "possession_frames": p["possession_frames"],
                "possession_share_pct": round(
                    p["possession_frames"] / total_frames * 100, 2
                )
                if total_frames > 0
                else 0.0,
                "total_distance_m": round(p["max_distance"], 2),
                "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "max_speed_kmh": round(float(np.max(speeds)), 2) if speeds else 0.0,
            }

        # ------------------------------------------------------------------
        # Team-level possession counts
        # ------------------------------------------------------------------
        t1_frames = int(np.sum(team_ball_control == 1))
        t2_frames = int(np.sum(team_ball_control == 2))
        total_possession_frames = t1_frames + t2_frames

        teams_stats: Dict[int, Dict[str, Any]] = {}
        for team_id in [1, 2]:
            team_players = {
                pid: s
                for pid, s in players_stats.items()
                if s["team"] == team_id
            }
            distances = [p["total_distance_m"] for p in team_players.values()]
            avg_speeds = [
                p["avg_speed_kmh"]
                for p in team_players.values()
                if p["avg_speed_kmh"] > 0
            ]
            pos_frames = t1_frames if team_id == 1 else t2_frames
            teams_stats[team_id] = {
                "team_id": team_id,
                "player_count": len(team_players),
                "possession_pct": round(
                    pos_frames / total_possession_frames * 100, 2
                )
                if total_possession_frames > 0
                else 0.0,
                "total_distance_m": round(sum(distances), 2),
                "avg_speed_kmh": round(float(np.mean(avg_speeds)), 2)
                if avg_speeds
                else 0.0,
            }

        # ------------------------------------------------------------------
        # Match-level summary
        # ------------------------------------------------------------------
        duration_s = total_frames / self.fps
        all_speeds = [
            p["avg_speed_kmh"]
            for p in players_stats.values()
            if p["avg_speed_kmh"] > 0
        ]
        match_stats: Dict[str, Any] = {
            "total_frames": total_frames,
            "duration_seconds": round(duration_s, 2),
            "duration_str": _fmt_duration(duration_s),
            "total_players_detected": len(players_stats),
            "team1_possession_pct": teams_stats.get(1, {}).get("possession_pct", 0.0),
            "team2_possession_pct": teams_stats.get(2, {}).get("possession_pct", 0.0),
            "overall_avg_speed_kmh": round(float(np.mean(all_speeds)), 2)
            if all_speeds
            else 0.0,
        }

        return {
            "match": match_stats,
            "players": players_stats,
            "teams": teams_stats,
        }
