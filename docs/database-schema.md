# Database Schema Documentation

## Overview

The drone security agent uses SQLite for structured storage with the following tables:

```
frames          ──┬── objects       (one-to-many)
                  ├── relates to ──→ events
                  
alerts          ──→ events         (many-to-one)
```

## Table Schemas

### `frames`
Stores video frame metadata and VLM captions.

```sql
CREATE TABLE frames (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    video_id TEXT,
    frame_path TEXT,
    caption TEXT,
    telemetry_json TEXT
);
```

**Fields:**
- `id`: Unique frame identifier (UUID)
- `ts`: Frame timestamp
- `video_id`: Source video identifier
- `frame_path`: Path to frame file (relative to data/)
- `caption`: VLM-generated caption describing frame content
- `telemetry_json`: JSON with drone telemetry (altitude, position, etc.)

**Indexes:**
- `idx_frames_ts`: For time-based queries

**Example:**
```python
store.store_frame(
    frame_id="frame_20260418_114500",
    video_id="video_1",
    frame_path="frames/video_1/frame_114500.jpg",
    caption="Two people walking near main gate"
)
```

---

### `objects`
Stores detected objects per frame.

```sql
CREATE TABLE objects (
    id INTEGER PRIMARY KEY,
    frame_id TEXT REFERENCES frames(id),
    class TEXT,
    confidence REAL,
    bbox TEXT,
    track_id INTEGER
);
```

**Fields:**
- `id`: Auto-incrementing object detection ID
- `frame_id`: Reference to the frame
- `class`: Object class (person, car, truck, drone, etc.)
- `confidence`: Detection confidence score (0-1)
- `bbox`: JSON with bounding box `{x, y, w, h}`
- `track_id`: Cross-frame tracking ID for trajectories

**Indexes:**
- `idx_objects_class`: For class-based queries

**Example:**
```python
store.store_object(
    frame_id="frame_20260418_114500",
    class_name="person",
    confidence=0.92,
    bbox={"x": 150, "y": 200, "w": 50, "h": 120},
    track_id=42
)
```

---

### `events`
Stores security events (high-level summaries).

```sql
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    type TEXT,
    severity TEXT,
    description TEXT,
    frame_ids TEXT
);
```

**Fields:**
- `id`: Unique event ID (UUID)
- `start_ts`: Event start time
- `end_ts`: Event end time (may be NULL for ongoing events)
- `type`: Event type (intrusion, crowd, unauthorized_vehicle, etc.)
- `severity`: Level (low, medium, high, critical)
- `description`: Human-readable description
- `frame_ids`: JSON array of associated frame IDs

**Indexes:**
- `idx_events_type`: For event type queries

**Example:**
```python
store.store_event(
    event_id="event_20260418_114500",
    event_type="person_detection",
    severity="medium",
    description="Unauthorized person detected in main gate area",
    frame_ids=["frame_20260418_114500", "frame_20260418_114501"],
    start_ts=datetime.now(timezone.utc)
)
```

---

### `alerts`
Stores alert delivery records.

```sql
CREATE TABLE alerts (
    id TEXT PRIMARY KEY,
    event_id TEXT REFERENCES events(id),
    ts TIMESTAMP,
    channel TEXT,
    message TEXT,
    acked INTEGER DEFAULT 0
);
```

**Fields:**
- `id`: Alert ID (UUID)
- `event_id`: Reference to triggering event
- `ts`: Alert send time
- `channel`: Delivery channel (console, email, webhook, slack)
- `message`: Alert message content
- `acked`: Acknowledgment flag (0=unacked, 1=acked)

**Example:**
```python
store.store_alert(
    alert_id="alert_20260418_114501",
    event_id="event_20260418_114500",
    channel="console",
    message="SECURITY ALERT: Unauthorized person in main gate"
)
```

---

## Query Patterns

### Common Queries

**Get all people detected in last hour:**
```python
from datetime import datetime, timezone, timedelta
hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
now = datetime.now(timezone.utc)
people_frames = store.query_frames_by_time(hour_ago, now)
for frame in people_frames:
    objects = store.get_frame_objects(frame['id'])
    people = [o for o in objects if o['class'] == 'person']
```

**Get high-severity events:**
```python
critical_events = store.get_events_by_severity('critical', hours=24)
for event in critical_events:
    frame_ids = json.loads(event['frame_ids'])
    print(f"Event: {event['description']} ({len(frame_ids)} frames)")
```

**Get unacknowledged alerts:**
```python
unacked = store.get_unacknowledged_alerts()
for alert in unacked:
    print(f"Alert: {alert['message']}")
    store.acknowledge_alert(alert['id'])
```

**Find repeat offenders (tracking):**
```python
recent_people = store.get_objects_by_class('person', limit=1000)
track_ids = {}
for obj in recent_people:
    track_id = obj['track_id']
    if track_id:
        track_ids.setdefault(track_id, []).append(obj)
        
# Objects with many detections = repeat presence
for track_id, detections in track_ids.items():
    if len(detections) > 20:
        print(f"Repeat detection: track_id {track_id} ({len(detections)} detections)")
```

---

## Working with SQLiteStore

### Initialization

```python
from src.memory import SQLiteStore

# Create/connect to database
store = SQLiteStore(db_path="data/security_memory.db")
```

### Storing Data

```python
# Frame
store.store_frame(
    frame_id="frame_001",
    video_id="video_1",
    frame_path="frames/video_1/001.jpg",
    caption="Scene description",
    telemetry={"altitude": 100, "lat": 37.123, "lon": -122.456}
)

# Objects
obj_id = store.store_object(
    frame_id="frame_001",
    class_name="person",
    confidence=0.95,
    bbox={"x": 10, "y": 20, "w": 50, "h": 100},
    track_id=5
)

# Event
store.store_event(
    event_id="event_001",
    event_type="intrusion",
    severity="high",
    description="Unauthorized access detected",
    frame_ids=["frame_001", "frame_002"]
)

# Alert
store.store_alert(
    alert_id="alert_001",
    event_id="event_001",
    channel="console",
    message="SECURITY ALERT: Intrusion detected!"
)
```

### Querying Data

```python
# Get frame
frame = store.get_frame("frame_001")

# Get detections in frame
objects = store.get_frame_objects("frame_001")

# Get all people
people = store.get_objects_by_class("person", limit=100)

# Get events by type
intrusions = store.get_events_by_type("intrusion", limit=50)

# Get high-severity events
critical = store.get_events_by_severity("critical", hours=24)

# Get frames with people
people_frames = store.get_frames_with_detections("person", limit=50)

# Get time range
from datetime import datetime, timezone, timedelta
start = datetime.now(timezone.utc) - timedelta(hours=1)
end = datetime.now(timezone.utc)
recent_frames = store.query_frames_by_time(start, end)

# Get unacknowledged alerts
unacked = store.get_unacknowledged_alerts()
```

### Cleanup

```python
# Delete old events
deleted = store.cleanup_old_events(retention_days=30)

# Close database
store.close()
```

---

## Backup and Maintenance

### Backup

```bash
# Copy SQLite database file
cp data/security_memory.db data/security_memory.backup.db
```

### Vacuuming (Optimize)

```python
cursor = store.connection.cursor()
cursor.execute("VACUUM")
store.connection.commit()
```

### Integrity Check

```python
cursor = store.connection.cursor()
result = cursor.execute("PRAGMA integrity_check").fetchone()
print(result[0])  # "ok" if healthy
```

---

## Performance Tips

1. **Indexes**: Queries on `ts`, `class`, and `type` are optimized with indexes
2. **Batch Operations**: Insert multiple records in a transaction for speed
3. **Cleanup**: Run `cleanup_old_events()` regularly to manage database size
4. **Retention**: Configure `retention_days` in settings to control storage
5. **JSON Fields**: Parse (`json.loads`) and serialize (`json.dumps`) as needed

---

## JSON Field Examples

### Frame Telemetry

```json
{
  "altitude": 100.5,
  "latitude": 37.7749,
  "longitude": -122.4194,
  "battery": 85,
  "speed": 5.2,
  "heading": 180,
  "gimbal_pitch": -45
}
```

### Object Bounding Box

```json
{
  "x": 150,
  "y": 200,
  "w": 50,
  "h": 120
}
```

### Event Frame IDs

```json
["frame_001", "frame_002", "frame_003"]
```
