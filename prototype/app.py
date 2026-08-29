"""
SIH 2026 — Interactive Vision-Language Assistant for Remote Sensing
Flask Backend API & Multi-Modal Orchestration Server
"""

import os
import io
import base64
import json
import time
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image

from model import (
    classify_image,
    detect_temporal_changes,
    analyze_optical_sar_joint,
    get_sample_images,
    get_qa_response,
    CATEGORIES,
    DATASET_PATH,
)

# ─────────────────────────────────────────────────────────────────
# Gemini Vision API (Optional with automatic graceful fallback)
# ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_AVAILABLE = False
gemini_model = None

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
        print("[OK] Gemini Vision API enabled")
    else:
        print("[INFO] No Gemini API key provided — using rule-based & spectral reasoning engine (fully offline)")
except Exception as e:
    print(f"[WARN] Gemini API not available: {e}")

# ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Session storage (in-memory, keyed by session_id from frontend)
sessions: dict = {}

# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    POST /api/classify
    Body: multipart/form-data with 'image' file + optional 'session_id'
    Returns: full classification result with spectral overlays, grounding boxes, and trace
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        session_id = request.form.get("session_id", "default")

        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Classify & extract multimodal overlays
        result = classify_image(pil_image)

        # Store in session for follow-up conversational Q&A
        sessions[session_id] = {
            "classification": result,
            "image_b64": base64.b64encode(img_bytes).decode(),
            "history": [],
            "last_active": time.time(),
        }

        return jsonify({
            "success": True,
            "classification": result,
            "gemini_available": GEMINI_AVAILABLE,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST /api/chat
    Body JSON: { session_id, question }
    Returns: { answer, source, intent, execution_trace }
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Empty question"}), 400

        session = sessions.get(session_id)
        if not session:
            return jsonify({"error": "No image loaded. Please upload a satellite image first."}), 400

        classification = session["classification"]
        history = session.get("history", [])

        answer = None
        source = "analysis-engine"
        intent = "vqa_query"
        execution_trace = []

        # Try Gemini first if available
        if GEMINI_AVAILABLE and session.get("image_b64"):
            try:
                import google.generativeai as genai
                img_bytes = base64.b64decode(session["image_b64"])
                pil_img = Image.open(io.BytesIO(img_bytes))

                prompt = (
                    f"You are SatQuery AI, an expert remote sensing vision-language assistant for SIH 2026. "
                    f"The image has been pre-analyzed with spectral statistics:\n"
                    f"- Class: {classification['label']} ({classification['confidence']}% confidence)\n"
                    f"- Spectral Stats: R={classification['stats']['mean_r']}, G={classification['stats']['mean_g']}, B={classification['stats']['mean_b']}\n"
                    f"- Vegetation Proxy (NDVI): {classification['stats']['ndvi_proxy']}\n"
                    f"- Water Index Proxy (NDWI): {classification['stats']['ndwi_proxy']}\n"
                    f"- Albedo/Brightness: {classification['stats']['brightness']}/255\n\n"
                    f"User question: {question}\n\n"
                    f"Provide an evidence-grounded, professional, concise response in 2-4 sentences using Markdown formatting."
                )

                t0 = time.time()
                response = gemini_model.generate_content([prompt, pil_img])
                lat = round((time.time() - t0) * 1000, 1)
                answer = response.text
                source = "gemini"
                execution_trace = [
                    {"step": 1, "tool": "Query & Context Formatter", "status": "Completed", "latency_ms": 12.0, "info": f"Formatted multispectral prompt for {classification['label']}"},
                    {"step": 2, "tool": "Gemini 1.5 Flash Vision API", "status": "Completed", "latency_ms": lat, "info": "Generated grounded multimodal response"}
                ]
            except Exception as ge:
                print(f"[WARN] Gemini error: {ge}")
                answer = None

        # Fallback to intelligent rule-based / analysis engine
        if not answer:
            qa_res = get_qa_response(question, classification)
            answer = qa_res["answer"]
            source = qa_res["source"]
            intent = qa_res.get("intent", "vqa_query")
            execution_trace = qa_res.get("execution_trace", [])

        # Update session history
        history.append({"role": "user", "text": question})
        history.append({"role": "assistant", "text": answer, "source": source})
        sessions[session_id]["history"] = history[-20:]

        return jsonify({
            "success": True,
            "answer": answer,
            "source": source,
            "intent": intent,
            "execution_trace": execution_trace,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/change-detection", methods=["POST"])
def api_change_detection():
    """
    POST /api/change-detection
    Accepts: 'image_t1' and 'image_t2' multipart files
    Or: 't1_cat', 't1_file', 't2_cat', 't2_file' for gallery samples
    Returns: bi-temporal change metrics, difference heatmap, and synthesis report
    """
    try:
        img1 = None
        img2 = None

        # Direct file uploads
        if "image_t1" in request.files and "image_t2" in request.files:
            file1 = request.files["image_t1"]
            file2 = request.files["image_t2"]
            img1 = Image.open(io.BytesIO(file1.read())).convert("RGB")
            img2 = Image.open(io.BytesIO(file2.read())).convert("RGB")
        # Dataset sample reference
        elif request.is_json:
            data = request.get_json()
            t1_cat = data.get("t1_cat")
            t1_file = data.get("t1_file")
            t2_cat = data.get("t2_cat")
            t2_file = data.get("t2_file")

            if t1_cat and t1_file and t2_cat and t2_file:
                p1 = os.path.join(DATASET_PATH, t1_cat, t1_file)
                p2 = os.path.join(DATASET_PATH, t2_cat, t2_file)
                if os.path.exists(p1) and os.path.exists(p2):
                    img1 = Image.open(p1).convert("RGB")
                    img2 = Image.open(p2).convert("RGB")

        if img1 is None or img2 is None:
            return jsonify({"error": "Please provide both Observation T1 (Before) and T2 (After) images."}), 400

        result = detect_temporal_changes(img1, img2)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/optical-sar", methods=["POST"])
def api_optical_sar():
    """
    POST /api/optical-sar
    Accepts: 'optical_image' and optional 'sar_image' multipart files
    Or: 'category' and 'filename' dataset reference
    Returns: cross-modal fusion composite, SAR simulated backscatter, and synthesis
    """
    try:
        opt_img = None
        sar_img = None

        if "optical_image" in request.files:
            opt_file = request.files["optical_image"]
            opt_img = Image.open(io.BytesIO(opt_file.read())).convert("RGB")
            if "sar_image" in request.files:
                sar_file = request.files["sar_image"]
                sar_img = Image.open(io.BytesIO(sar_file.read())).convert("RGB")
        elif request.is_json:
            data = request.get_json()
            cat = data.get("category")
            filename = data.get("filename")
            if cat and filename:
                p = os.path.join(DATASET_PATH, cat, filename)
                if os.path.exists(p):
                    opt_img = Image.open(p).convert("RGB")

        if opt_img is None:
            return jsonify({"error": "Please provide an optical satellite image."}), 400

        result = analyze_optical_sar_joint(opt_img, sar_img)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """GET /api/samples?category=desert&count=12"""
    category = request.args.get("category", "desert")
    count = int(request.args.get("count", 12))

    if category not in CATEGORIES:
        return jsonify({"error": "Invalid category"}), 400

    files = get_sample_images(category, count)
    return jsonify({
        "success": True,
        "category": category,
        "files": files,
        "total": len(files),
    })


@app.route("/api/categories", methods=["GET"])
def api_categories():
    """GET /api/categories — list all categories with metadata and image counts"""
    result = []
    for key, meta in CATEGORIES.items():
        folder = os.path.join(DATASET_PATH, key)
        count = 0
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        result.append({
            "key": key,
            "label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "desc": meta.get("desc", ""),
            "count": count,
        })
    return jsonify({"success": True, "categories": result})


@app.route("/dataset/<category>/<filename>")
def dataset_image(category, filename):
    """Serve dataset images"""
    folder = os.path.join(DATASET_PATH, category)
    return send_from_directory(folder, filename)


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running",
        "system": "SatQuery AI — Remote Sensing Vision-Language Assistant",
        "hackathon": "Smart India Hackathon 2026",
        "gemini_available": GEMINI_AVAILABLE,
        "dataset_path": DATASET_PATH,
        "categories": list(CATEGORIES.keys()),
        "features": [
            "Scene Classification & Spectral Profiling",
            "NDVI / NDWI / CIR Spectral Overlays",
            "Visual Grounding & Hotspot Bounding Boxes",
            "Bi-Temporal Change Detection",
            "Optical + SAR Cross-Modal Fusion",
            "Auditable Agent Execution Tracing",
            "Natural Language Vision-Language Q&A"
        ]
    })


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SatQuery AI — Interactive Vision-Language Assistant (SIH 2026)")
    print("="*65)
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Gemini Vision API: {'Enabled' if GEMINI_AVAILABLE else 'Offline Spectral / Rule-Based'}")
    print(f"  Web Interface: http://localhost:5000")
    print("="*65 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
