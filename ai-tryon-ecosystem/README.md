# 🛍️ AI-Driven Virtual Try-On Ecosystem

**COMSATS University Islamabad — BS Business Data Analytics FYP 2024–25**

A full-stack AI application for virtual try-on of clothes, makeup, accessories, and personalized size recommendations.

---

## 📁 Project Structure

```
ai-tryon-ecosystem/
├── frontend/                 ← Open these HTML files in browser
│   ├── index.html            ← Home / Landing page
│   ├── pages/
│   │   ├── clothes.html      ← Module 1: Clothes try-on
│   │   ├── makeup.html       ← Module 2: Makeup try-on
│   │   ├── accessories.html  ← Module 3: Accessories try-on
│   │   └── size.html         ← Module 4: Size recommender
│   ├── css/shared.css
│   └── js/api.js
│
├── backend/                  ← Flask Python server
│   ├── app.py                ← START HERE — main server file
│   ├── config.py
│   ├── requirements.txt      ← pip install -r requirements.txt
│   ├── routes/               ← API endpoints
│   ├── models/               ← Database models
│   └── utils/
│
├── ai_engine/                ← All AI/ML code
│   ├── clothes/tryon_processor.py
│   ├── makeup/makeup_processor.py
│   ├── accessories/accessory_processor.py
│   └── size/size_predictor.py
│
└── data/
    ├── store_inventory/demo_products.csv
    └── uploads/              ← User photos saved here
```

---

## 🚀 Setup & Run (Step by Step)

### Step 1 — Install Python (if not installed)
Download Python 3.10+ from https://python.org

### Step 2 — Open Terminal in project folder
```bash
cd ai-tryon-ecosystem
```

### Step 3 — Create virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r backend/requirements.txt
```
⚠️ This may take 5–10 minutes (downloading OpenCV, MediaPipe, etc.)

### Step 5 — Set up environment file
```bash
# Windows:
copy .env.example .env

# Mac/Linux:
cp .env.example .env
```

### Step 6 — Train the size model (optional but recommended)
```bash
cd ai_engine/size
python train_model.py
cd ../..
```

### Step 7 — Start the Flask server
```bash
python backend/app.py
```
You should see:
```
* Running on http://127.0.0.1:5000
```

### Step 8 — Seed demo data (run once)
Open browser and visit: http://localhost:5000/api/inventory/seed

OR run this in another terminal:
```bash
curl -X POST http://localhost:5000/api/inventory/seed
```

### Step 9 — Open the app
Open `frontend/index.html` in your browser.

---

## 🖥️ How to Use Each Module

### Module 1 — Clothes Try-On
1. Open `frontend/pages/clothes.html`
2. Upload a **full body photo** (standing, clear background works best)
3. Click any clothing item
4. Click "Try It On"

### Module 2 — Makeup Try-On
1. Open `frontend/pages/makeup.html`
2. Upload a **face photo** (frontal, well-lit)
3. Choose lipstick, eyeshadow, blush or foundation
4. Adjust intensity with slider
5. Click "Apply Makeup"

### Module 3 — Accessories Try-On
1. Open `frontend/pages/accessories.html`
2. Upload your photo
3. Filter by type (glasses, hat, earrings, necklace)
4. Click an accessory
5. Click "Try Accessory On"

### Module 4 — Size Recommender
1. Open `frontend/pages/size.html`
2. Enter your body measurements in cm
3. Click "Find My Size"
4. Get size recommendation with fit advice

---

## 🔧 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/inventory/seed | Seed demo products |
| GET | /api/inventory/products | Get all products |
| GET | /api/clothes/products | Get clothing items |
| POST | /api/clothes/tryon | Try on a garment |
| GET | /api/makeup/products | Get makeup items |
| POST | /api/makeup/apply | Apply makeup |
| GET | /api/accessories/products | Get accessories |
| POST | /api/accessories/tryon | Try on accessory |
| POST | /api/size/recommend | Get size recommendation |
| GET | /api/size/chart | View size chart |

---

## 📦 Key Libraries Used

| Library | Purpose |
|---------|---------|
| Flask | Backend web server & REST API |
| MediaPipe | Face/body landmark detection (468 face points) |
| OpenCV | Image processing, color blending, overlays |
| scikit-learn | ML model for size prediction |
| SQLAlchemy | Database ORM |
| Pillow | Image manipulation |

---

## 🎓 FYP Information

- **University:** COMSATS University Islamabad
- **Program:** BS Business Data Analytics
- **Semester:** 7th (Final Year)
- **Project:** AI-Driven Virtual Try-On Ecosystem
- **Modules:** 4 (Clothes, Makeup, Accessories, Size)

---

## 📧 Common Issues

**"No module named mediapipe"**
→ Run: `pip install mediapipe`

**"No face detected"**
→ Use a clear, well-lit frontal face photo. Avoid side angles.

**"Backend not running"**
→ Make sure `python backend/app.py` is running in terminal

**Products not loading**
→ Visit http://localhost:5000/api/inventory/seed first
