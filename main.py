import os, time, base64, json, traceback
import requests, cv2, numpy as np

# ---------------- CONFIG ----------------
RADAR_URL = os.getenv("RADAR_URL", "")
LINE_TOKEN = os.getenv("LINE_TOKEN", "")
MY_LINE_USER_ID = os.getenv("MY_LINE_USER_ID", "")
IMGBB_KEY = os.getenv("IMGBB_KEY", "")

ROI = (738, 626, 1730, 1580)

MIN_PIXELS = 150
COOLDOWN_SEC = 30 * 60
STATE_FILE = "last_alert.json"

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

    content_type = r.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise ValueError(f"RADAR_URL points to a Webpage (HTML), not an Image file! URL used: {RADAR_URL}")

    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not decode image from RADAR_URL. Error: {e}")

    return img


def detect(img):
    clean_img = img.copy()
    h, w = clean_img.shape[:2]

    clean_img[:, 0:65] = (0, 0, 0)
    clean_img[0:130, 0:130] = (0, 0, 0)
    clean_img[int(h * 0.85):h, int(w * 0.75):w] = (0, 0, 0)

    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
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
                x, y, w_box, h_box = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x + w_box, y + h_box), BOX_COLOR[name], 2)

    if ROI:
        cv2.rectangle(out, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)
    return result, out


def upload_imgbb(img, max_retries=3):
    if not IMGBB_KEY:
        raise ValueError("IMGBB_KEY secret is missing!")

    h, w = img.shape[:2]
    scale = 1600 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Failed to encode image")

    payload = {
        "image": base64.b64encode(buf).decode(),
        "expiration": 86400
    }

    # เพิ่มระบบพยายามส่งซ้ำ (Retry) สูงสุด 3 ครั้งหากเจอเน็ตหลุดหรือ Timeout
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Uploading image to ImgBB (Attempt {attempt}/{max_retries})...")
            r = requests.post(
                "https://api.imgbb.com/1/upload",
                params={"key": IMGBB_KEY},
                data=payload,
                timeout=60,
            )
            if r.ok:
                res_json = r.json()
                if res_json.get("success"):
                    url = res_json["data"]["url"]
                    print(f"ImgBB Upload Success: {url}")
                    return url
            last_err = f"imgbb {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
            print(f"ImgBB upload attempt {attempt} failed: {e}")
        
        if attempt < max_retries:
            time.sleep(5)  # หน่วงเวลา 5 วินาทีก่อนลองใหม่อีกครั้ง

    raise RuntimeError(f"All ImgBB retries failed. Last error: {last_err}")


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
    if not r.ok:
        raise RuntimeError(f"LINE {r.status_code}: {r.text[:300]}")


def in_cooldown():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return time.time() - json.load(f)["t"] < COOLDOWN_SEC
    except Exception:
        pass
    return False


def notify_error(err):
    try:
        msg = f"⚠️ TMD Radar Alert ผิดพลาด\n{type(err).__name__}: {str(err)[:400]}"
        line_push(msg)
    except Exception as e2:
        print(f"Could not send error notification: {e2}")


def main():
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

    img = fetch_image()
    counts, annotated = detect(img)
    strong = counts["yellow"] + counts["orange"] + counts["red"]
    print(f"Detected pixels: {counts}")

    if not TEST_MODE and (strong < MIN_PIXELS or in_cooldown()):
        print("No significant rain detected or in cooldown.")
        return

    if TEST_MODE and strong == 0:
        level = "🧪 [TEST] ทดสอบระบบ (ไม่พบกลุ่มฝน)"
    elif counts["red"] >= MIN_PIXELS // 3:
        level = "🔴 ฝนหนักมาก"
    elif counts["orange"] > 0:
        level = "🟠 ฝนหนัก"
    elif counts["yellow"] > 0:
        level = "🟡 ฝนปานกลาง"
    else:
        level = "🟢 ฝนตกเล็กน้อย / ปกติ"

    text = (f"{level} ในพื้นที่เรดาร์\n"
            f"เหลือง {counts['yellow']} | ส้ม {counts['orange']} | แดง {counts['red']} px")

    try:
        image_url = upload_imgbb(annotated)
    except Exception as e:
        print(f"Image upload failed: {e}")
        image_url = None
        text += "\n(อัปโหลดรูปไม่สำเร็จ)"

    line_push(text, image_url)

    if not TEST_MODE:
        with open(STATE_FILE, "w") as f:
            json.dump({"t": time.time()}, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        notify_error(e)
        raise
