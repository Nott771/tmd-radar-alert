import os, time, base64, json
import requests, cv2, numpy as np

# ---------------- CONFIG ----------------
RADAR_URL = os.getenv("RADAR_URL", "")
LINE_TOKEN = os.getenv("LINE_TOKEN", "")
MY_LINE_USER_ID = os.getenv("MY_LINE_USER_ID", "")
IMGBB_KEY = os.getenv("IMGBB_KEY", "")

# พื้นที่ที่สนใจ (พิกเซล x1,y1,x2,y2 ของรูปเรดาร์) -> ตั้ง None = ตรวจทั้งรูป
ROI = None  # เช่น (420, 380, 560, 520)

MIN_PIXELS = 150          # กี่พิกเซลถึงจะแจ้งเตือน
COOLDOWN_SEC = 30 * 60    # แจ้งซ้ำได้ทุกกี่วินาที
STATE_FILE = "last_alert.json"

# ช่วงสี HSV (OpenCV: H 0-179)
RANGES = {
    "yellow": [((20, 120, 150), (35, 255, 255))],
    "orange": [((10, 120, 150), (19, 255, 255))],
    "red":    [((0, 120, 120), (9, 255, 255)), ((170, 120, 120), (179, 255, 255))],
}
BOX_COLOR = {"yellow": (0, 255, 255), "orange": (0, 165, 255), "red": (0, 0, 255)}
# -----------------------------------------


def fetch_image():
    if not RADAR_URL or "http" not in RADAR_URL:
        raise ValueError("RADAR_URL is invalid or missing in GitHub Secrets!")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(RADAR_URL, timeout=30, headers=headers)
    r.raise_for_status()

    # ตรวจสอบว่า URL ที่ดึงมาเป็นรูปภาพหรือไม่
    content_type = r.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise ValueError(f"RADAR_URL points to a Webpage (HTML), not an Image file! URL used: {RADAR_URL}")

    # ลอง decode ด้วย OpenCV
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    
    # ถ้า OpenCV อ่านไม่ได้ ลองเปิดด้วย PIL (กรณีเป็น GIF Animation)
    if img is None:
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not decode image from RADAR_URL. Make sure it's a direct link to .png/.jpg/.gif! Error: {e}")

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
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
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
    if not IMGBB_KEY:
        raise ValueError("IMGBB_KEY secret is missing!")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    r = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_KEY, "image": base64.b64encode(buf).decode(), "expiration": 86400},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]["url"]


def line_push(text, image_url=None):
    if not LINE_TOKEN or not MY_LINE_USER_ID:
        raise ValueError("LINE_TOKEN or MY_LINE_USER_ID secret is missing!")
    msgs = [{"type": "text", "text": text}]
    if image_url:
        msgs.append({"type": "image", "originalContentUrl": image_url, "previewImageUrl": image_url})
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {LINE_TOKEN}"},
        json={"to": MY_LINE_USER_ID, "messages": msgs},
        timeout=30,
    )
    r.raise_for_status()


def in_cooldown():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return time.time() - json.load(f)["t"] < COOLDOWN_SEC
    except Exception:
        pass
    return False


def main():
    img = fetch_image()
    counts, annotated = detect(img)
    strong = counts["yellow"] + counts["orange"] + counts["red"]
    print(f"Detected pixels: {counts}")

    if strong < MIN_PIXELS or in_cooldown():
        print("No significant rain detected or in cooldown.")
        return

    level = "🔴 ฝนหนักมาก" if counts["red"] >= MIN_PIXELS // 3 else "🟠 ฝนหนัก" if counts["orange"] else "🟡 ฝนปานกลาง"
    text = (f"{level} ตรวจพบกลุ่มฝนในพื้นที่เรดาร์\n"
            f"เหลือง {counts['yellow']} | ส้ม {counts['orange']} | แดง {counts['red']} px")
    line_push(text, upload_imgbb(annotated))
    
    with open(STATE_FILE, "w") as f:
        json.dump({"t": time.time()}, f)


if __name__ == "__main__":
    main()
