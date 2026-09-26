from flask import Flask, render_template, request, jsonify
import re
import json
import base64
import requests
import csv
import io
import os
from google import genai

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

STRONG_KEYWORDS = [
    "delivery", "delivered", "delivery driver", "delivery rider", "courier",
    "delivery app", "jahez", "hungerstation", "talabat", "mrsool", "the chefz",
    "مندوب", "توصيل", "دليڤري", "هنقرستيشن", "جاهز", "مرسول",
    "تطبيق توصيل", "طلبات",
]


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FFa-zA-Z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_reviews(reviews_raw):
    total = len(reviews_raw)
    removed_empty = 0
    removed_short = 0
    removed_duplicate = 0
    removed_noise = 0
    cleaned = []
    seen = set()
    for r in reviews_raw:
        if not r or not r.strip():
            removed_empty += 1
            continue
        c = clean_text(r)
        if not c:
            removed_noise += 1
            continue
        if len(c) < 5:
            removed_short += 1
            continue
        if c in seen:
            removed_duplicate += 1
            continue
        cleaned.append(c)
        seen.add(c)
    report = {
        "total_before": total,
        "total_after": len(cleaned),
        "removed_empty": removed_empty,
        "removed_short": removed_short,
        "removed_duplicate": removed_duplicate,
        "removed_noise": removed_noise,
        "removed_total": total - len(cleaned),
    }
    return cleaned, report


def parse_csv(file_content, filter_name=""):
    reader = csv.DictReader(io.StringIO(file_content))
    rows = list(reader)
    if not rows:
        return None

    keys = list(rows[0].keys())
    name_col = next((c for c in keys if c.lower() in ["name","restaurant","اسم المطعم","title"]), None)
    review_col = next((c for c in keys if c.lower() in ["review","text","تقييم","النص","snippet","review_text"]), None)
    address_col = next((c for c in keys if c.lower() in ["address","العنوان","location"]), None)
    rating_col = next((c for c in keys if c.lower() in ["rating","التقييم","totalscore","score"]), None)
    reviews_count_col = next((c for c in keys if c.lower() in ["reviews_count","reviewscount","عدد التقييمات"]), None)
    photo_col = next((c for c in keys if c.lower() in ["photo","image","صورة","imageurl","photo_url"]), None)

    if filter_name:
        rows = [r for r in rows if filter_name.lower() in (r.get(name_col,"") or "").lower()]
    if not rows:
        return None

    first = rows[0]
    name = clean_text(first.get(name_col,"")) if name_col else ""
    address = clean_text(first.get(address_col,"")) if address_col else ""
    rating = first.get(rating_col,"") if rating_col else ""
    reviews_count = first.get(reviews_count_col,"") if reviews_count_col else str(len(rows))

    reviews = []
    photo_urls = []
    seen_photos = set()

    for row in rows:
        if review_col:
            review = row.get(review_col,"")
            if review:
                reviews.append(review)
        if photo_col:
            photo_val = row.get(photo_col,"") or ""
            for part in photo_val.split(","):
                part = part.strip()
                if part.startswith("http") and part not in seen_photos:
                    photo_urls.append(part)
                    seen_photos.add(part)
        for val in row.values():
            if val and isinstance(val, str) and val.strip().startswith("https://lh") and val.strip() not in seen_photos:
                photo_urls.append(val.strip())
                seen_photos.add(val.strip())

    return name, address, rating, reviews_count, reviews, photo_urls


def parse_json(file_content, filter_name=""):
    data = json.loads(file_content)
    if isinstance(data, list):
        if filter_name:
            data = [d for d in data if filter_name.lower() in (d.get("name","") or d.get("title","") or "").lower()]
        if not data:
            return None
        item = data[0]
        reviews_raw = []
        photo_urls = []
        for d in data:
            for r in d.get("reviews",[]) or []:
                if isinstance(r,str): reviews_raw.append(r)
                elif isinstance(r,dict):
                    text = r.get("text","") or r.get("review","") or r.get("snippet","")
                    if text: reviews_raw.append(text)
            for p in d.get("photos",[]) or d.get("images",[]) or []:
                if isinstance(p,str) and p.startswith("http"): photo_urls.append(p)
                elif isinstance(p,dict):
                    url = p.get("url","") or p.get("imageUrl","")
                    if url: photo_urls.append(url)
    else:
        item = data
        reviews_raw = []
        for r in data.get("reviews",[]) or []:
            if isinstance(r,str): reviews_raw.append(r)
            elif isinstance(r,dict):
                text = r.get("text","") or r.get("review","") or r.get("snippet","")
                if text: reviews_raw.append(text)
        photo_urls = []
        for p in data.get("photos",[]) or data.get("images",[]) or []:
            if isinstance(p,str) and p.startswith("http"): photo_urls.append(p)
            elif isinstance(p,dict):
                url = p.get("url","") or p.get("imageUrl","")
                if url: photo_urls.append(url)

    name = clean_text(item.get("name","") or item.get("title","") or "")
    address = clean_text(item.get("address","") or "")
    rating = item.get("rating","") or item.get("totalScore","") or ""
    reviews_count = item.get("reviews_count","") or item.get("reviewsCount","") or str(len(reviews_raw))
    return name, address, rating, reviews_count, reviews_raw, photo_urls


def nlp_analyze(reviews):
    mentions = 0
    reviews_with_signals = 0
    found = set()
    for r in reviews:
        lower = r.lower()
        has_signal = False
        for kw in STRONG_KEYWORDS:
            if kw.lower() in lower:
                mentions += 1
                has_signal = True
                found.add(kw)
        if has_signal:
            reviews_with_signals += 1
    density = reviews_with_signals / max(len(reviews), 1)
    return {
        "delivery_mentions_count": mentions,
        "reviews_with_delivery_signals": reviews_with_signals,
        "found_keywords": list(found),
        "keyword_density": round(density, 3),
        "total_reviews_analyzed": len(reviews),
    }


def image_url_to_part(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    mime = r.headers.get("Content-Type", "image/jpeg")
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(r.content).decode("utf-8"),
        }
    }


def gemini_analyze(name, reviews, photo_urls):
    reviews_text = "\n".join(reviews)
    parts = [{
        "text": f"""
أنت محلل بيانات متخصص في اقتصاد المنصات (Platform Economy).
مهمتك: تحليل مدى اعتماد هذا المطعم على تطبيقات التوصيل بناءً على التقييمات النصية والصور.

اسم المطعم: {name}

التقييمات:
{reviews_text}

قواعد التحليل:
- اعتمد فقط على الأدلة الموجودة في النصوص أو الصور.
- كلمات دلالة قوية: delivery, delivered, courier, delivery app, jahez, hungerstation, talabat, mrsool, مندوب, توصيل, جاهز, مرسول, تطبيق توصيل, هنقرستيشن, طلبات.
- لا تعتبر هذه وحدها دليلاً: order, ordered, take away, pickup, counter, food bag.
- photos_with_delivery_signals: عدد الصور التي تحتوي دليل بصري قوي مثل حقيبة توصيل أو يزي مندوب أو دراجة توصيل.
- delivery_objects: قائمة بالأشياء البصرية المكتشفة.

أجب فقط بـ JSON بدون أي شرح أو backticks:
{{"delivery_keywords":[],"delivery_mentions_count":0,"reviews_with_delivery_signals":0,"photos_with_delivery_signals":0,"delivery_objects":[],"summary":"","risk_level":"Low","platform_dependency_score":0}}
"""
    }]

    for url in photo_urls:
        try:
            parts.append(image_url_to_part(url))
        except Exception:
            pass

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[{"parts": parts}],
        )
        raw = response.text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "delivery_keywords": [],
        "delivery_mentions_count": 0,
        "reviews_with_delivery_signals": 0,
        "photos_with_delivery_signals": 0,
        "delivery_objects": [],
        "summary": "تعذّر الاتصال بـ Gemini.",
        "risk_level": "Low",
        "platform_dependency_score": 0,
    }


def calc_metrics(merged):
    dm = merged.get("delivery_mentions_count", 0)
    rws = merged.get("reviews_with_delivery_signals", 0)
    pws = merged.get("photos_with_delivery_signals", 0)
    pds = min(merged.get("platform_dependency_score", 0), 10)
    keyword_density = merged.get("keyword_density", 0)

    digital_activity_score = round(
         (rws * 1.2) + (pws * 1.8) + (keyword_density * 5), 2
    )
    platform_dependency_index = round(pds / 10, 2)
    estimated_gig_workers = round(1 + (rws * 1.2) + (pws * 2.0) + (keyword_density * 5))
    registered_workers = max(1, round(estimated_gig_workers * 0.35))
    activity_gap = estimated_gig_workers - registered_workers

    if activity_gap >= 15:
        gap_level = "Large Gap"
    elif activity_gap >= 7:
        gap_level = "Medium Gap"
    else:
        gap_level = "Low / No Gap"

    return {
        "digital_activity_score": digital_activity_score,
        "platform_dependency_index": platform_dependency_index,
        "estimated_gig_workers": estimated_gig_workers,
        "registered_workers": registered_workers,
        "activity_gap": activity_gap,
        "gap_level": gap_level,
        "keyword_density": round(keyword_density, 3),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "لم يتم رفع ملف"}), 400
    filename = file.filename.lower()
    content = file.read().decode("utf-8-sig")
    try:
        if filename.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            name_col = next((c for c in (rows[0].keys() if rows else []) if c.lower() in ["name","restaurant","اسم المطعم","title"]), None)
            names = list(dict.fromkeys([r.get(name_col,"").strip() for r in rows if r.get(name_col,"").strip()])) if name_col else []
            return jsonify({"names": names[:100], "total_rows": len(rows)})
        elif filename.endswith(".json"):
            data = json.loads(content)
            if isinstance(data, list):
                names = list(dict.fromkeys([(d.get("name","") or d.get("title","") or "").strip() for d in data if (d.get("name","") or d.get("title",""))]))
            else:
                names = [(data.get("name","") or data.get("title","")).strip()]
            return jsonify({"names": names[:100], "total_rows": len(data) if isinstance(data,list) else 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    filter_name = request.form.get("filter_name", "").strip()
    if not file:
        return jsonify({"error": "لم يتم رفع ملف"}), 400

    filename = file.filename.lower()
    content = file.read().decode("utf-8-sig")

    try:
        if filename.endswith(".csv"):
            result = parse_csv(content, filter_name)
        elif filename.endswith(".json"):
            result = parse_json(content, filter_name)
        else:
            return jsonify({"error": "صيغة غير مدعومة، استخدم CSV أو JSON"}), 400
    except Exception as e:
        return jsonify({"error": f"خطأ في قراءة الملف: {str(e)}"}), 400

    if not result:
        return jsonify({"error": "لم يتم العثور على بيانات"}), 400

    name, address, rating, reviews_count, reviews_raw, photo_urls = result
    reviews, cleaning_report = clean_reviews(reviews_raw)

    if not reviews:
        return jsonify({"error": "لم يتم العثور على تقييمات في الملف"}), 400

    nlp = nlp_analyze(reviews)
    ai = gemini_analyze(name, reviews, photo_urls)

    keyword_density = nlp["keyword_density"]
    photos_with_signals = min(ai.get("photos_with_delivery_signals", 0), len(photo_urls))

    if keyword_density > 0.30 or photos_with_signals > 5:
        risk_level = "High"
    elif keyword_density > 0.10 or photos_with_signals > 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    merged = {
        "delivery_mentions_count": max(nlp["delivery_mentions_count"], ai.get("delivery_mentions_count", 0)),
        "reviews_with_delivery_signals": max(nlp["reviews_with_delivery_signals"], ai.get("reviews_with_delivery_signals", 0)),
        "photos_with_delivery_signals": photos_with_signals,
        "platform_dependency_score": min(ai.get("platform_dependency_score", 0), 10),
        "risk_level": risk_level,
        "delivery_keywords": list(set(nlp["found_keywords"] + ai.get("delivery_keywords", []))),
        "delivery_objects": ai.get("delivery_objects", []),
        "summary": ai.get("summary", ""),
        "keyword_density": nlp["keyword_density"],
        "total_reviews_analyzed": nlp["total_reviews_analyzed"],
        "total_photos_analyzed": len(photo_urls),
    }

    obs = calc_metrics(merged)

    return jsonify({
        "name": name,
        "address": address,
        "rating": rating,
        "reviews_count": reviews_count,
        "analysis": merged,
        "observatory_metrics": obs,
        "cleaning_report": cleaning_report,
    })


if __name__ == "__main__":
    app.run(debug=True)
