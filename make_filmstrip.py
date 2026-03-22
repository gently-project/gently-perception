"""
Generate a filmstrip viewer for manual annotation verification.

Creates:
1. Thumbnail JPEG for each timepoint per embryo
2. A self-contained HTML page with filmstrip + annotations

Usage:
    python make_filmstrip.py
    Then open data/filmstrip/viewer.html in a browser.
"""

import base64
import json
from pathlib import Path

# Reuse the testset's projection code
from benchmark.testset import (
    _discover_volumes,
    _load_volume,
    _create_three_view_image,
)
from benchmark.ground_truth import GroundTruth

DATA_DIR = Path(__file__).parent / "data"
VOLUMES_DIR = DATA_DIR / "volumes"
GT_PATH = DATA_DIR / "ground_truth" / "59799c78.json"
OUTPUT_DIR = DATA_DIR / "filmstrip"

STAGES = ["early", "bean", "comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"]
STAGE_COLORS = {
    "early": "#4CAF50",
    "bean": "#8BC34A",
    "comma": "#CDDC39",
    "1.5fold": "#FFC107",
    "2fold": "#FF9800",
    "pretzel": "#FF5722",
    "hatching": "#9C27B0",
    "hatched": "#673AB7",
}


def generate_thumbnails():
    """Generate thumbnail images for all timepoints."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thumbs_dir = OUTPUT_DIR / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)

    gt = GroundTruth.from_json(GT_PATH)
    all_volumes = _discover_volumes(VOLUMES_DIR)

    embryo_data = {}

    for embryo_id in sorted(all_volumes.keys()):
        if embryo_id not in gt.transitions:
            continue

        volumes = all_volumes[embryo_id]
        print(f"Processing {embryo_id}: {len(volumes)} timepoints...")

        timepoints = []
        for tp_idx, vol_path in enumerate(volumes):
            # Generate thumbnail
            thumb_path = thumbs_dir / f"{embryo_id}_t{tp_idx:03d}.jpg"

            if not thumb_path.exists():
                try:
                    vol = _load_volume(vol_path)
                    img_b64 = _create_three_view_image(vol, max_dim=400)
                    # Save as file
                    import io
                    from PIL import Image
                    img_bytes = base64.b64decode(img_b64)
                    thumb_path.write_bytes(img_bytes)
                except Exception as e:
                    print(f"  Error at T{tp_idx}: {e}")
                    continue

            stage = gt.get_stage_at(embryo_id, tp_idx)
            timepoints.append({
                "tp": tp_idx,
                "stage": stage,
                "thumb": f"thumbs/{embryo_id}_t{tp_idx:03d}.jpg",
            })

        embryo_data[embryo_id] = {
            "transitions": gt.transitions[embryo_id],
            "timepoints": timepoints,
        }
        print(f"  Done: {len(timepoints)} thumbnails")

    return embryo_data


def generate_html(embryo_data):
    """Generate the HTML viewer."""
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Embryo Filmstrip Viewer</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }
h1 { text-align: center; margin-bottom: 5px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }
.embryo { margin-bottom: 40px; border: 1px solid #333; border-radius: 8px; padding: 15px; }
.embryo h2 { margin: 0 0 10px 0; }
.transitions { font-size: 13px; color: #aaa; margin-bottom: 10px; }
.transitions span { margin-right: 12px; }
.filmstrip { display: flex; overflow-x: auto; gap: 2px; padding: 10px 0; }
.frame { flex-shrink: 0; text-align: center; cursor: pointer; position: relative; }
.frame img { width: 120px; height: auto; border: 3px solid transparent; border-radius: 4px; display: block; }
.frame img:hover { border-color: #fff; }
.frame .label { font-size: 10px; padding: 2px 4px; border-radius: 2px; margin-top: 2px; }
.frame .tp { font-size: 9px; color: #666; }
.frame.transition-start img { border-color: #ff0 !important; border-width: 3px; }
.legend { display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 13px; }
.legend-color { width: 16px; height: 16px; border-radius: 3px; }
.modal { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.9); z-index: 100; justify-content: center; align-items: center; }
.modal.active { display: flex; }
.modal img { max-width: 90vw; max-height: 90vh; }
.modal .info { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 10px 20px; border-radius: 8px; font-size: 18px; }
.modal .nav { position: fixed; top: 50%; font-size: 48px; cursor: pointer; color: #fff; user-select: none; }
.modal .nav:hover { color: #ff0; }
.modal .nav.prev { left: 20px; }
.modal .nav.next { right: 20px; }
</style>
</head>
<body>
<h1>Embryo Filmstrip Viewer</h1>
<p class="subtitle">Click any frame to enlarge. Yellow border = stage transition. Arrow keys to navigate.</p>
<div class="legend">
"""
    for stage, color in STAGE_COLORS.items():
        html += f'<div class="legend-item"><div class="legend-color" style="background:{color}"></div>{stage}</div>\n'

    html += '</div>\n'

    # Modal for enlarged view
    html += """
<div class="modal" id="modal" onclick="closeModal()">
    <div class="info" id="modal-info"></div>
    <img id="modal-img" src="">
    <div class="nav prev" onclick="event.stopPropagation(); navModal(-1)">&#9664;</div>
    <div class="nav next" onclick="event.stopPropagation(); navModal(1)">&#9654;</div>
</div>
"""

    for embryo_id, data in sorted(embryo_data.items()):
        html += f'<div class="embryo">\n<h2>{embryo_id}</h2>\n'
        html += '<div class="transitions">'
        for stage, tp in sorted(data["transitions"].items(), key=lambda x: x[1]):
            color = STAGE_COLORS.get(stage, "#888")
            html += f'<span style="color:{color}">&#9632; {stage} @ T{tp}</span>'
        html += '</div>\n'
        html += '<div class="filmstrip">\n'

        transition_tps = set(data["transitions"].values())

        for tp_data in data["timepoints"]:
            tp = tp_data["tp"]
            stage = tp_data["stage"] or "unknown"
            color = STAGE_COLORS.get(stage, "#888")
            is_transition = tp in transition_tps
            cls = "frame transition-start" if is_transition else "frame"

            html += f'''<div class="{cls}" onclick="openModal('{tp_data['thumb']}', '{embryo_id} T{tp} — {stage}', this)">
  <img src="{tp_data['thumb']}" loading="lazy">
  <div class="label" style="background:{color};color:#000">{stage}</div>
  <div class="tp">T{tp}</div>
</div>\n'''

        html += '</div>\n</div>\n'

    html += """
<script>
let currentFrames = [];
let currentIdx = 0;

function openModal(src, info, el) {
    document.getElementById('modal').classList.add('active');
    document.getElementById('modal-img').src = src;
    document.getElementById('modal-info').textContent = info;

    // Build frame list from parent filmstrip
    const strip = el.parentElement;
    currentFrames = Array.from(strip.querySelectorAll('.frame'));
    currentIdx = currentFrames.indexOf(el);
}

function closeModal() {
    document.getElementById('modal').classList.remove('active');
}

function navModal(dir) {
    currentIdx = Math.max(0, Math.min(currentFrames.length - 1, currentIdx + dir));
    const frame = currentFrames[currentIdx];
    const img = frame.querySelector('img');
    const label = frame.querySelector('.label');
    const tp = frame.querySelector('.tp');
    document.getElementById('modal-img').src = img.src;
    document.getElementById('modal-info').textContent = tp.textContent + ' — ' + label.textContent;
}

document.addEventListener('keydown', (e) => {
    if (document.getElementById('modal').classList.contains('active')) {
        if (e.key === 'ArrowLeft') navModal(-1);
        else if (e.key === 'ArrowRight') navModal(1);
        else if (e.key === 'Escape') closeModal();
    }
});
</script>
</body>
</html>"""

    output_path = OUTPUT_DIR / "viewer.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"\nViewer saved to: {output_path}")
    print("Open in browser to view.")


if __name__ == "__main__":
    print("Generating filmstrip viewer...")
    embryo_data = generate_thumbnails()
    generate_html(embryo_data)
    print("Done!")
