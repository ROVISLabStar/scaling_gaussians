"""
Viser-based comparison viewer for DDVS experiments.
====================================================

Replays saved frames from all methods side by side in a web viewer,
showing convergence curves and frame-by-frame comparison.

Usage:
    python -m gs_vs_ddvs.viser.comparison_viewer --log_dir gs_vs_ddvs/logs/test_run3
    Then open http://localhost:8080

Author: Youssef ALJ (UM6P)
"""

import argparse
import os
import glob
import time
import numpy as np
import cv2
import viser


METHOD_COLORS = {
    'original':  (255, 80, 80),
    'ddvs':      (0, 180, 0),
    'inflated':  (80, 80, 255),
    'pgm_vs':    (255, 165, 0),
}

METHOD_LABELS = {
    'original':  'Original PVS',
    'ddvs':      'DDVS',
    'inflated':  'Inflated 3DGS',
    'pgm_vs':    'PGM-VS',
}


def load_method_frames(log_dir, method):
    """Load all saved frames for a method, sorted by iteration."""
    method_dir = os.path.join(log_dir, method)
    if not os.path.isdir(method_dir):
        return []
    frames = sorted(glob.glob(os.path.join(method_dir, "frame_*.png")))
    return frames


def main():
    p = argparse.ArgumentParser(description="DDVS Comparison Viewer")
    p.add_argument("--log_dir", required=True, help="Directory with method subdirs")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    # Discover methods
    methods = []
    method_frames = {}
    for name in ['original', 'ddvs', 'inflated', 'pgm_vs']:
        frames = load_method_frames(args.log_dir, name)
        if frames:
            methods.append(name)
            method_frames[name] = frames
            print(f"[{name}] {len(frames)} frames")

    if not methods:
        print(f"No frames found in {args.log_dir}/")
        return

    max_frames = max(len(method_frames[m]) for m in methods)

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible", control_width="large")
    print(f"[Viser] http://localhost:{args.port}")

    # GUI controls
    with server.gui.add_folder("Playback"):
        slider = server.gui.add_slider("Frame", min=0, max=max_frames - 1,
                                       step=1, initial_value=0)
        play_btn = server.gui.add_button("Play")
        speed = server.gui.add_slider("Speed (fps)", min=1, max=30,
                                      step=1, initial_value=5)

    # Method toggles
    method_toggles = {}
    with server.gui.add_folder("Methods"):
        for m in methods:
            label = METHOD_LABELS.get(m, m)
            method_toggles[m] = server.gui.add_checkbox(label, initial_value=True)

    # Image panels — one per method
    panels = {}
    for i, m in enumerate(methods):
        label = METHOD_LABELS.get(m, m)
        with server.gui.add_folder(label):
            panels[m] = server.gui.add_image(image=np.zeros((100, 300, 3), dtype=np.uint8))

    def update_display(frame_idx):
        for m in methods:
            if not method_toggles[m].value:
                panels[m].image = np.zeros((100, 300, 3), dtype=np.uint8)
                continue
            frames = method_frames[m]
            idx = min(frame_idx, len(frames) - 1)
            img = cv2.imread(frames[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            panels[m].image = img

    @slider.on_update
    def _(_):
        update_display(slider.value)

    playing = [False]

    @play_btn.on_click
    def _(_):
        playing[0] = not playing[0]
        play_btn.name = "Pause" if playing[0] else "Play"
        if playing[0]:
            import threading
            def _play():
                while playing[0] and slider.value < max_frames - 1:
                    slider.value = slider.value + 1
                    update_display(slider.value)
                    time.sleep(1.0 / speed.value)
                playing[0] = False
                play_btn.name = "Play"
            threading.Thread(target=_play, daemon=True).start()

    # Show first frame
    update_display(0)

    print("[Viewer] Running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
