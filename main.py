"""
TMD radar -> detect yellow/orange/red (rain) -> alert via LINE Messaging API
pip install requests opencv-python numpy
รันทุก 10 นาทีด้วย cron / Task Scheduler / GitHub Actions
"""
import os, time, base64, json
import requests, cv2, numpy as np

# ---------------- CONFIG ----------------
RADAR_URL = os.getenv("RADAR_URL", "PUT_TMD_RADAR_IMAGE_URL_HERE")  # ดู URL จากหน้าเรดาร์ TMD (F12 > Network > Img)
LINE_TOKEN = os.getenv("LINE_TOKEN", "")      # Channel access token (long-lived)
LINE_TO = os.getenv("LINE_TO", "")            # userId หรือ groupId
IMGBB_KEY = os.getenv("IMGBB_KEY", "")        # https://api.imgbb.com/

# พื้นที่ที่สนใจ (พิกเซล x1,y1,x2,y2 ของรูปเรดาร์) -> ปรับให้ครอบชลบุรี/บางละมุง
# ตั้ง None = ตรวจทั้งรูป
ROI = None  # เช่น (420, 380, 560, 520)

MIN_PIXELS = 150          # กี่พิกเซลถึงจะแจ้งเตือน (ปรับตามขนาดรูป)
COOLDOWN_SEC = 30 * 60    # แจ้งซ้ำได้ทุกกี่วินาที
STATE_FILE = "last_alert.json"

# ช่วงสี HSV (OpenCV: H 0-179) — ต้องจูนกับ legend จริงของ TMD
RANGES = {
    "yellow": [((20, 120, 150), (35, 255, 255))],
    "orange": [((10, 120, 150), (19, 255, 255))],
    "red":    [((0, 120, 120), (9, 255, 255)), ((170, 120, 120), (179, 255, 255))],
}
BOX_COLOR = {"yellow": (0, 255, 255), "orange": (0, 165, 255), "red": (0, 0, 255)}
# -----------------------------------------


def fetch_image():
    r = requests.get(RADAR_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:  # เผื่อเป็น GIF
        from PIL import Image
        import io
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(r.content)).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros(img.shape[:2], np.uint8)
    if ROI:
        x1, y1, x2, y2 = ROI
        roi_mask[y1:y2, x1:x2] = 255
    else:
        roi_mask[:] = 255

    result, out = {}, img.copy()
    kernel = np.ones((3, 3), np.uint8)
    for name, rngs in RANGES.items():
        m = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in rngs:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)  # ตัด noise เส้นแผนที่/ตัวอักษร
        m &= roi_mask
        result[name] = int(cv2.countNonZero(m))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) >= 20:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR[name], 2)
    if ROI:
        cv2.rectangle(out, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
    return result, out


def upload_imgbb(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    r = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_KEY, "image": base64.b64encode(buf).decode(), "expiration": 86400},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]["url"]


def line_push(text, image_url=None):
    msgs = [{"type": "text", "text": text}]
    if image_url:
        msgs.append({"type": "image", "originalContentUrl": image_url, "previewImageUrl": image_url})
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {LINE_TOKEN}"},
        json={"to": LINE_TO, "messages": msgs},
        timeout=30,
    )
    r.raise_for_status()


def in_cooldown():
    try:
        return time.time() - json.load(open(STATE_FILE))["t"] < COOLDOWN_SEC
    except Exception:
        return False


def main():
    img = fetch_image()
    counts, annotated = detect(img)
    strong = counts["yellow"] + counts["orange"] + counts["red"]
    print(counts)

    if strong < MIN_PIXELS or in_cooldown():
        return

    level = "🔴 ฝนหนักมาก" if counts["red"] >= MIN_PIXELS // 3 else "🟠 ฝนหนัก" if counts["orange"] else "🟡 ฝนปานกลาง"
    text = (f"{level} ตรวจพบกลุ่มฝนในพื้นที่เรดาร์\n"
            f"เหลือง {counts['yellow']} | ส้ม {counts['orange']} | แดง {counts['red']} px")
    line_push(text, upload_imgbb(annotated))
    json.dump({"t": time.time()}, open(STATE_FILE, "w"))


if __name__ == "__main__":
    main()
