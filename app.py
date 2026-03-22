"""
app.py  –  Football Analysis Streamlit Dashboard
-------------------------------------------------
Entry point for the web UI.  Run with:

    streamlit run app.py

Requires the model weights at  models/best.pt
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── project imports ───────────────────────────────────────────────────────────
from analytics_collector import AnalyticsCollector
from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from report_generator import (
    REPORTLAB_AVAILABLE,
    generate_pdf_report,
    save_stats_csv,
    save_stats_json,
)
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
VIDEO_FPS = 24.0

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── minimal CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .section-divider { margin: 1rem 0; border-top: 2px solid #e0e8ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _run_pipeline(input_video_path: str, work_dir: str) -> tuple[str, dict]:
    """
    Execute the complete analysis pipeline on ``input_video_path``.

    Returns
    -------
    (output_video_path, stats_dict)
    """
    output_avi = os.path.join(work_dir, "output_annotated.avi")
    output_mp4 = os.path.join(work_dir, "output_annotated.mp4")

    bar = st.progress(0, text="📂 Reading video frames…")

    # ── 1. Read video ─────────────────────────────────────────────────────────
    video_frames = read_video(input_video_path)
    bar.progress(8, text="🔍 Detecting & tracking objects…")

    # ── 2. Object tracking ────────────────────────────────────────────────────
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    tracker.add_position_to_tracks(tracks)
    bar.progress(25, text="📷 Estimating camera movement…")

    # ── 3. Camera motion compensation ─────────────────────────────────────────
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_movement = cam_est.get_camera_movement(video_frames, read_from_stub=False)
    cam_est.add_adjust_positions_to_tracks(tracks, cam_movement)
    bar.progress(38, text="🗺️  Applying perspective transform…")

    # ── 4. View transform → real-world coordinates ───────────────────────────
    view_tf = ViewTransformer()
    view_tf.add_transformed_position_to_tracks(tracks)
    bar.progress(48, text="🎾 Interpolating ball positions…")

    # ── 5. Ball interpolation ─────────────────────────────────────────────────
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    bar.progress(55, text="⚡ Computing speed & distance…")

    # ── 6. Speed / distance metrics ───────────────────────────────────────────
    speed_dist = SpeedAndDistance_Estimator()
    speed_dist.add_speed_and_distance_to_tracks(tracks)
    bar.progress(65, text="🎽 Assigning player teams…")

    # ── 7. Team assignment ────────────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    for frame_num, player_track in enumerate(tracks["players"]):
        for pid, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], pid
            )
            tracks["players"][frame_num][pid]["team"] = team
            tracks["players"][frame_num][pid]["team_color"] = team_assigner.team_colors[team]
    bar.progress(78, text="🏃 Assigning ball possession…")

    # ── 8. Ball possession assignment ────────────────────────────────────────
    ball_assigner = PlayerBallAssigner()
    team_ball_control: list[int] = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks["players"][frame_num][assigned]["has_ball"] = True
            team_ball_control.append(
                int(tracks["players"][frame_num][assigned]["team"])
            )
        else:
            # Carry last known possession; default to 0 if first frame
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control_arr = np.array(team_ball_control, dtype=int)
    bar.progress(85, text="🎨 Drawing annotations…")

    # ── 9. Render annotated frames ────────────────────────────────────────────
    out_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control_arr)
    out_frames = cam_est.draw_camera_movement(out_frames, cam_movement)
    speed_dist.draw_speed_and_distance(out_frames, tracks)
    bar.progress(90, text="💾 Saving video…")

    # ── 10. Save AVI (original codec) ────────────────────────────────────────
    save_video(out_frames, output_avi)

    # ── 11. Convert to MP4 for in-browser playback ───────────────────────────
    _convert_to_mp4(output_avi, output_mp4)
    bar.progress(95, text="📊 Collecting analytics…")

    # ── 12. Analytics ─────────────────────────────────────────────────────────
    collector = AnalyticsCollector(fps=VIDEO_FPS)
    stats = collector.process_tracks(tracks, team_ball_control_arr)
    bar.progress(100, text="✅ Done!")

    return output_mp4, stats


def _convert_to_mp4(src_avi: str, dst_mp4: str) -> None:
    """Re-encode AVI → MP4 (H.264/mp4v) for browser compatibility."""
    cap = cv2.VideoCapture(src_avi)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_mp4, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────


def _render_stats(stats: dict) -> None:
    """Render match / team / player stats inside the Streamlit app."""
    import pandas as pd  # local import — not required at module load time

    match = stats.get("match", {})
    teams = stats.get("teams", {})
    players = stats.get("players", {})

    # ── Match summary ─────────────────────────────────────────────────────────
    st.subheader("📋 Match Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⏱ Duration", match.get("duration_str", "N/A"))
    c2.metric("👥 Players", match.get("total_players_detected", 0))
    c3.metric("🔵 Team 1 Possession", f"{match.get('team1_possession_pct', 0.0):.1f}%")
    c4.metric("🔴 Team 2 Possession", f"{match.get('team2_possession_pct', 0.0):.1f}%")
    c5.metric("⚡ Overall Avg Speed", f"{match.get('overall_avg_speed_kmh', 0.0):.2f} km/h")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Team stats ────────────────────────────────────────────────────────────
    st.subheader("🏟️ Team Statistics")
    team_rows = []
    for tid in sorted(teams.keys()):
        t = teams[tid]
        team_rows.append(
            {
                "Team": f"Team {tid}",
                "Players Tracked": t.get("player_count", 0),
                "Possession %": f"{t.get('possession_pct', 0.0):.1f}%",
                "Total Distance (m)": f"{t.get('total_distance_m', 0.0):.2f}",
                "Avg Speed (km/h)": f"{t.get('avg_speed_kmh', 0.0):.2f}",
            }
        )
    if team_rows:
        st.dataframe(
            pd.DataFrame(team_rows), use_container_width=True, hide_index=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Player stats ──────────────────────────────────────────────────────────
    st.subheader("👤 Player Statistics")
    player_rows = []
    for pid in sorted(players.keys()):
        p = players[pid]
        player_rows.append(
            {
                "Player ID": pid,
                "Team": p.get("team", "?"),
                "Distance (m)": p.get("total_distance_m", 0.0),
                "Avg Speed (km/h)": p.get("avg_speed_kmh", 0.0),
                "Max Speed (km/h)": p.get("max_speed_kmh", 0.0),
                "Possession Frames": p.get("possession_frames", 0),
                "Possession %": p.get("possession_share_pct", 0.0),
                "Tracked Frames": p.get("tracked_frames", 0),
            }
        )
    if player_rows:
        df = pd.DataFrame(player_rows)
        st.dataframe(
            df.sort_values("Distance (m)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("⚽ AI Football Analysis Dashboard")
    st.markdown(
        "Upload a match video clip to run the full computer-vision pipeline — "
        "multi-player tracking, team assignment, speed & distance metrics, "
        "and ball possession analytics."
    )

    # ── Pre-flight check ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"**Model weights not found** at `{MODEL_PATH}`.  "
            "Place `best.pt` inside the `models/` directory, then refresh."
        )
        st.stop()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("---")
    uploaded = st.file_uploader(
        "📁 Upload a football match video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    if uploaded is None:
        st.info("👆 Upload a video above to get started.")
        return

    st.success(f"Video loaded: **{uploaded.name}** ({uploaded.size / 1_048_576:.1f} MB)")

    # ── Analyse button ────────────────────────────────────────────────────────
    if st.button("🚀 Analyse Video", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as work_dir:
            # Write uploaded bytes to a temp file
            suffix = Path(uploaded.name).suffix or ".mp4"
            input_path = os.path.join(work_dir, f"input{suffix}")
            with open(input_path, "wb") as fh:
                fh.write(uploaded.getbuffer())

            try:
                output_mp4_path, stats = _run_pipeline(input_path, work_dir)

                # ── Persist results in session state (survives reruns) ─────────
                with open(output_mp4_path, "rb") as fh:
                    st.session_state["video_bytes"] = fh.read()

                st.session_state["stats"] = stats

                # ── Generate downloadable files inside the same tempdir ────────
                json_p = os.path.join(work_dir, "football_stats.json")
                csv_p = os.path.join(work_dir, "football_stats.csv")
                pdf_p = os.path.join(work_dir, "football_report.pdf")

                save_stats_json(stats, json_p)
                save_stats_csv(stats, csv_p)
                pdf_ok = generate_pdf_report(stats, pdf_p)

                with open(json_p, "rb") as fh:
                    st.session_state["json_bytes"] = fh.read()
                with open(csv_p, "rb") as fh:
                    st.session_state["csv_bytes"] = fh.read()
                if pdf_ok:
                    with open(pdf_p, "rb") as fh:
                        st.session_state["pdf_bytes"] = fh.read()
                else:
                    st.session_state["pdf_bytes"] = None

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                raise

    # ── Results section ───────────────────────────────────────────────────────
    if "video_bytes" not in st.session_state:
        return

    st.markdown("---")
    st.success("✅ Analysis complete!")

    tab_video, tab_stats, tab_download = st.tabs(
        ["🎬 Annotated Video", "📊 Analytics", "⬇️ Downloads"]
    )

    with tab_video:
        st.video(st.session_state["video_bytes"])

    with tab_stats:
        _render_stats(st.session_state["stats"])

    with tab_download:
        st.subheader("Download Analysis Files")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                label="📄 Stats (JSON)",
                data=st.session_state["json_bytes"],
                file_name="football_stats.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📊 Stats (CSV)",
                data=st.session_state["csv_bytes"],
                file_name="football_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col3:
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    label="📋 Report (PDF)",
                    data=st.session_state["pdf_bytes"],
                    file_name="football_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("Install `reportlab` to enable PDF export.")
        with col4:
            st.download_button(
                label="🎬 Annotated Video",
                data=st.session_state["video_bytes"],
                file_name="annotated_match.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        # Inline stats JSON viewer
        with st.expander("🔎 Inspect raw stats (JSON)"):
            import json as _json

            st.code(
                _json.dumps(st.session_state["stats"], indent=2),
                language="json",
            )


if __name__ == "__main__":
    main()
