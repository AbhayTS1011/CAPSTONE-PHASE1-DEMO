#!/usr/bin/env bash
# Runs the measured backend test suite (backend must be running on 127.0.0.1:8000). Results -> test_results/
set -u
PY=backend/.venv/Scripts/python
OUT=test_results
for v in test_videos/scene-*.mp4 ../github/input/PSJ_01_Singapore_Queenstown_Input.mp4; do
  n=$(basename "$v" .mp4)
  echo "== unpaced: $n"; $PY tools/smoke_test_backend.py "$v" --no-pace --out "$OUT/run_unpaced_$n.json" 2>&1 | tail -1
done
echo "== paced + pause test: scene-0796"; $PY tools/smoke_test_backend.py test_videos/scene-0796_singapore-queenstown_CAM_FRONT_234f.mp4 --pause-test --out "$OUT/run_paced_pause_scene-0796.json" 2>&1 | tail -1
echo "== long video (1170 frames), unpaced"; $PY tools/smoke_test_backend.py test_videos/long_queenstown_x5_1170f.mp4 --no-pace --out "$OUT/run_unpaced_long_x5.json" 2>&1 | tail -1
