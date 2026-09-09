"""
Configuration and constants for LipRead Studio.
Manages dataset paths, word lists (multilingual presets & custom loaders),
audio/video capture settings, themes, and quality control thresholds.
"""

from pathlib import Path
import json
import csv
import logging

# ============================================================
# APPLICATION DIRECTORIES
# ============================================================

APP_DIR = Path(__file__).resolve().parent

DATASET_DIR = APP_DIR / "dataset"
VIDEO_ROOT = DATASET_DIR / "videos"
AUDIO_ROOT = DATASET_DIR / "audio"
LIPS_ROOT = DATASET_DIR / "lips"
METADATA_ROOT = DATASET_DIR / "metadata"
REPORTS_ROOT = DATASET_DIR / "reports"
CHECKPOINTS_ROOT = DATASET_DIR / "checkpoints"
EXPORTS_ROOT = DATASET_DIR / "exports"

LOG_DIR = APP_DIR / "logs"

for folder in [
    VIDEO_ROOT,
    AUDIO_ROOT,
    LIPS_ROOT,
    METADATA_ROOT,
    REPORTS_ROOT,
    CHECKPOINTS_ROOT,
    EXPORTS_ROOT,
    LOG_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# WORD LISTS (MULTILINGUAL PRESETS)
# ============================================================

WORDS_ENGLISH_100 = [
    'Analyse', 'Acquire', 'Adult', 'At least', 'Average', 'Area',
    'Exam', 'Engineer', 'East', 'Earn', 'Entire', 'Equation',
    'Plan', 'Pharmacy', 'Perimeter', 'Pain', 'Please', 'Physics',
    'Bus', 'Bill', 'Bike', 'Bicycle', 'Brand', 'Biased',
    'Bureaucracy', 'Fan', 'Friend', 'Full', 'Function', 'Quadratic',
    'Geography', 'Group', 'Green', 'Room', 'Radius', 'Rural',
    'Computer', 'Car', 'Chemistry', 'Class', 'Circumference',
    'Crore', 'Control', 'Hall', 'Humidity', 'India', 'Jail',
    'Speed', 'Silence', 'Store', 'Student', 'Suicide', 'System',
    'Society', 'Knowledge', 'Kilo', 'Luck', 'Lakh', 'Length',
    'Dance', 'Danger', 'Drum', 'Diameter', 'Design', 'Dining',
    'Map', 'Marks', 'Machine', 'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday', 'Table', 'Turn',
    'Variety', 'Unique', 'Under', 'Estimate', 'Use', 'Salary',
    'Age', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November',
    'December', 'Semester', 'Today', 'Tomorrow', 'Yesterday'
]

WORDS_HINDI_100 = [
    'नमस्ते (Namaste)', 'भारत (Bharat)', 'विकास (Vikas)', 'ज्ञान (Gyan)',
    'समय (Samay)', 'परिवार (Parivar)', 'मित्र (Mitra)', 'सत्य (Satya)',
    'आशा (Asha)', 'प्रयास (Prayas)', 'सुरक्षा (Suraksha)', 'स्वास्थ्य (Swasthya)',
    'शिक्षा (Shiksha)', 'पुस्तक (Pustak)', 'विद्यालय (Vidyalay)', 'विज्ञान (Vigyan)',
    'गणित (Ganit)', 'कक्षा (Kaksha)', 'अध्यापक (Adhyapak)', 'छात्र (Chhatra)',
    'पानी (Pani)', 'भोजन (Bhojan)', 'फल (Phal)', 'सब्जी (Sabzi)',
    'घर (Ghar)', 'शहर (Shahar)', 'गाँव (Gaanv)', 'सड़क (Sadak)',
    'गाड़ी (Gaadi)', 'रेल (Rail)', 'विमान (Viman)', 'यात्रा (Yatra)',
    'दिन (Din)', 'रात (Raat)', 'सुबह (Subah)', 'शाम (Shaam)',
    'आज (Aaj)', 'कल (Kal)', 'हफ्ता (Hafta)', 'महीना (Mahina)',
    'साल (Saal)', 'ऋतु (Ritu)', 'मौसम (Mausam)', 'वर्षा (Varsha)',
    'धूप (Dhoop)', 'हवा (Hawa)', 'आकाश (Aakash)', 'पृथ्वी (Prithvi)',
    'नदी (Nadi)', 'समुद्र (Samudra)', 'पर्वत (Parvat)', 'जंगल (Jangal)',
    'सूरज (Suraj)', 'चाँद (Chaand)', 'तारा (Taara)', 'प्रकाश (Prakash)',
    'अंधकार (Andhakar)', 'ध्वनि (Dhwani)', 'भाषा (Bhasha)', 'शब्द (Shabd)',
    'वाक्य (Vakya)', 'गीत (Geet)', 'संगीत (Sangeet)', 'कला (Kala)',
    'रंग (Rang)', 'लाल (Laal)', 'नीला (Neela)', 'हरा (Hara)',
    'पीला (Peela)', 'सफेद (Safed)', 'काला (Kaala)', 'गुलाबी (Gulabi)',
    'सुंदर (Sundar)', 'कठिन (Kathin)', 'सरल (Saral)', 'नया (Naya)',
    'पुराना (Purana)', 'बड़ा (Bada)', 'छोटा (Chhota)', 'तेज (Tez)',
    'धीमा (Dheema)', 'सफल (Safal)', 'प्रसन्न (Prasann)', 'शांति (Shanti)',
    'उत्साह (Utsaah)', 'शक्ति (Shakti)', 'साहस (Saahas)', 'धैर्य (Dhairya)',
    'न्याय (Nyay)', 'धर्म (Dharm)', 'कर्म (Karm)', 'सेवा (Sewa)',
    'प्रेम (Prem)', 'आदर (Aadar)', 'धन्यवाद (Dhanyawad)', 'स्वागत (Swagat)',
    'अलविदा (Alvida)', 'प्रणाम (Pranam)', 'जीवन (Jeevan)', 'संसार (Sansaar)'
]

WORDS_ASSAMESE_50 = [
    'নমস্কাৰ (Nomoskar)', 'অসম (Oxom)', 'আই (Aai)', 'পিতা (Pita)',
    'পানী (Pani)', 'ভাত (Bhat)', 'ঘৰ (Ghor)', 'গাঁও (Gaaon)',
    'নগৰ (Nogor)', 'নৈ (Noi)', 'বতাহ (Botah)', 'বৰষুণ (Boroxun)',
    'সূৰ্য (Xurjyo)', 'জোন (Zun)', 'পোহৰ (Pohor)', 'আকাশ (Aakax)',
    'গছ (Gos)', 'ফুল (Phul)', 'মাটি (Mati)', 'পথ (Poth)',
    'যান (Zaan)', 'বিদ্যালয় (Bidyaloy)', 'কিতাপ (Kitap)', 'কলম (Kolom)',
    'কাম (Kaam)', 'সময় (Xomoy)', 'আজি (Aazi)', 'কাইলৈ (Kaaloi)',
    'ৰাতি (Raati)', 'পুৱা (Puwa)', 'মিতা (Mita)', 'বন্ধু (Bondhu)',
    'আনন্দ (Aanondo)', 'শান্তি (Xanti)', 'মৰম (Morom)', 'শ্রদ্ধা (Xraddha)',
    'ভাষা (Bhasa)', 'গীত (Geet)', 'সুৰ (Xur)', 'ৰং (Rong)',
    'ৰঙা (Ronga)', 'নীলা (Neela)', 'সেউজীয়া (Xeujia)', 'বগা (Boga)',
    'কলা (Kola)', 'ভাল (Bhal)', 'ধুনীয়া (Dhunia)', 'নতুন (Notun)',
    'পুৰণি (Puroni)', 'ধন্যবাদ (Dhonyobad)'
]

PRESET_WORD_LISTS = {
    "English (100 Words)": WORDS_ENGLISH_100,
    "Hindi (100 Words)": WORDS_HINDI_100,
    "Assamese (50 Words)": WORDS_ASSAMESE_50
}

def load_word_list_from_file(file_path: Path) -> list:
    """Load a custom word list from a CSV or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Word list file not found: {path}")

    words = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        words.append(item.strip())
                    elif isinstance(item, dict) and "word" in item:
                        words.append(str(item["word"]).strip())
    elif path.suffix.lower() in [".csv", ".txt"]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    word = row[0].strip()
                    if word and not word.lower().startswith("word"):
                        words.append(word)
    else:
        raise ValueError("Unsupported format. Use .csv, .txt, or .json")

    words = [w for w in words if w]
    if not words:
        raise ValueError("The provided file contains no valid words.")
    return words

# ============================================================
# CAMERA & AUDIO SETTINGS
# ============================================================

DEFAULT_CAMERA_INDEX = 0
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30.0

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

LIP_SIZE = 112
LIP_MARGIN_X = 0.35
LIP_MARGIN_Y = 0.60

MOUTH_LANDMARKS = [61, 291, 0, 17, 78, 308, 13, 14]

# Sound frequencies & durations
BEEP_FREQUENCY = 850
BEEP_DURATION = 0.12
START_FREQUENCY = 1200
START_DURATION = 0.30
STOP_FREQUENCY = 500
STOP_DURATION = 0.30

# ============================================================
# QUALITY CONTROL THRESHOLDS
# ============================================================

QC_PASS_THRESHOLD = 0.85
QC_REVIEW_THRESHOLD = 0.70
QC_FAILED_THRESHOLD = 0.70
MAX_RERECORD_ATTEMPTS = 2

FACE_WEIGHT = 0.30
LIP_WEIGHT = 0.70

# ============================================================
# THEME COLOR PALETTES
# ============================================================

THEME_PALETTES = {
    "cyberpunk": {
        "name": "Cyberpunk / Studio",
        "bg": "#070B14",
        "panel": "#0E1625",
        "panel_alt": "#111A28",
        "border": "#233450",
        "text": "#F4F8FF",
        "text_muted": "#71839A",
        "accent": "#55F6D2",
        "accent_secondary": "#7DD3FC",
        "highlight": "#38BDF8",
        "danger": "#FF647C",
        "danger_bg": "#7F1D1D",
        "warning": "#FDE68A",
        "warning_bg": "#4A3B13",
        "success": "#86EFAC",
        "btn_normal": "#172A43",
        "btn_active": "#263B59",
        "preview_bg": "#030712"
    },
    "research": {
        "name": "Research Mode (Minimalist)",
        "bg": "#11151C",
        "panel": "#1B222C",
        "panel_alt": "#212A36",
        "border": "#2D3A4B",
        "text": "#E2E8F0",
        "text_muted": "#8593A6",
        "accent": "#93C5FD",
        "accent_secondary": "#60A5FA",
        "highlight": "#3B82F6",
        "danger": "#F87171",
        "danger_bg": "#451214",
        "warning": "#FBBF24",
        "warning_bg": "#3B2E0A",
        "success": "#4ADE80",
        "btn_normal": "#263242",
        "btn_active": "#34445A",
        "preview_bg": "#0B0E14"
    },
    "demo": {
        "name": "Demo Mode (Vibrant HUD)",
        "bg": "#050811",
        "panel": "#0C1222",
        "panel_alt": "#121A30",
        "border": "#1E3054",
        "text": "#FFFFFF",
        "text_muted": "#8BA2C1",
        "accent": "#00F5D4",
        "accent_secondary": "#7B2CBF",
        "highlight": "#00BBF9",
        "danger": "#FF0054",
        "danger_bg": "#6B092B",
        "warning": "#FEE440",
        "warning_bg": "#594E07",
        "success": "#38B000",
        "btn_normal": "#172647",
        "btn_active": "#283E6E",
        "preview_bg": "#020409"
    },
    "light": {
        "name": "Clinical Light",
        "bg": "#F8FAFC",
        "panel": "#FFFFFF",
        "panel_alt": "#F1F5F9",
        "border": "#CBD5E1",
        "text": "#0F172A",
        "text_muted": "#64748B",
        "accent": "#0284C7",
        "accent_secondary": "#0369A1",
        "highlight": "#0EA5E9",
        "danger": "#DC2626",
        "danger_bg": "#FEE2E2",
        "warning": "#D97706",
        "warning_bg": "#FEF3C7",
        "success": "#16A34A",
        "btn_normal": "#E2E8F0",
        "btn_active": "#CBD5E1",
        "preview_bg": "#000000"
    }
}
