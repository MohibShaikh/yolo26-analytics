"""PPE safety monitoring — zones with required equipment classes.

This shows how yolo26-analytics can be configured for safety compliance
using a PPE-trained model. The zone analytics detect when workers enter
zones without required protective equipment.

Requires: A YOLO26 model fine-tuned on PPE classes (helmet, vest, goggles).
Public datasets available on Roboflow Universe.
"""

CONFIG = """
source:
  type: video_file
  path: construction_site.mp4

model:
  type: yolo26
  weights: yolo26n_ppe.pt
  confidence: 0.4

tracking:
  engine: bytetrack
  max_age: 30
  min_hits: 3

store:
  type: sqlite
  path: ./data/ppe_safety.db

zones:
  - name: "Hard Hat Zone"
    polygon: [[0,0], [640,0], [640,480], [0,480]]
    track_classes: [person, helmet, vest, goggles]
    analytics:
      - type: count
      - type: dwell
        alert_threshold: 5
      - type: entry_exit
    cooldown: 10

alerts:
  - type: console
  - type: webhook
    url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL

dashboard: true
"""

if __name__ == "__main__":
    from pathlib import Path

    config_path = Path("config_ppe.yaml")
    config_path.write_text(CONFIG)
    print(f"Config written to {config_path}")
    print("Run: y26a run --config config_ppe.yaml --dashboard")
