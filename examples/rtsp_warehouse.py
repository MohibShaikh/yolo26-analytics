"""RTSP camera + zones + Telegram alerts — warehouse scenario."""

# Using YAML config is the recommended approach for production setups.
# Save this as config_warehouse.yaml and run:
#   y26a run --config config_warehouse.yaml --dashboard

CONFIG = """
source:
  type: rtsp
  url: rtsp://192.168.1.100/stream

model:
  type: yolo26
  weights: yolo26n.pt
  confidence: 0.5

tracking:
  engine: bytetrack
  max_age: 30
  min_hits: 3

store:
  type: sqlite
  path: ./data/warehouse.db

zones:
  - name: "Loading Dock"
    polygon: [[100,100], [500,100], [500,400], [100,400]]
    track_classes: [person, forklift]
    analytics:
      - type: count
      - type: entry_exit
        direction: inward
      - type: dwell
        alert_threshold: 300
    cooldown: 30

  - name: "Restricted Area"
    polygon: [[600,50], [900,50], [900,300], [600,300]]
    track_classes: [person]
    analytics:
      - type: entry_exit
    cooldown: 10

alerts:
  - type: console
  - type: telegram
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
    filter:
      zones: ["Restricted Area"]
      event_types: ["entry"]

dashboard: true
"""

if __name__ == "__main__":
    from pathlib import Path
    from yolo26_analytics.core.pipeline import Pipeline

    config_path = Path("config_warehouse.yaml")
    config_path.write_text(CONFIG)
    print(f"Config written to {config_path}")
    print("Run: y26a run --config config_warehouse.yaml --dashboard")
