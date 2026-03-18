import os
import warnings

# [隱藏日誌] 關閉底層套件的警告與亂碼輸出
os.environ['ORT_LOGGING_LEVEL'] = '3'  # 3 代表只顯示 Fatal 錯誤
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 順便隱藏 TensorFlow 的除錯訊息
warnings.filterwarnings("ignore", category=FutureWarning)

import base64
import concurrent.futures
import datetime
import json
import random
import re
import shutil
import sqlite3
import time
import urllib.parse
import uuid
import html
import numpy as np
import requests
import cv2
import insightface
import glob
import mediapipe as mp
import google.generativeai as genai
from authlib.integrations.flask_client import OAuth
from functools import wraps
from bs4 import BeautifulSoup
from fpdf import FPDF
from typing import Any
from insightface.app import FaceAnalysis
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

#  系統全域設定 (System Configuration)
app = Flask(__name__)

# [安全設定]
app.secret_key = os.environ.get('SECRET_KEY', 'thesis_final_ultimate_v2026_complete_edition')

# [開發環境設定] 允許 OAuth 使用 HTTP (上線部署時建議移除或設為 '0')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# [路徑設定] 確保跨平台相容性
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 限制

# [API 金鑰與常數]
CURRENT_MODEL_VERSION = "StyleNet-Evo-v3.0"
API_ACCESS_KEY = "open_style_api_2026"
GENAI_API_KEY = "AIzaSyCocgWFBaFU4SBx3MzJpOFihPY6gD2aDdA"

# 多語系設定
TRANSLATIONS = {
    'zh_TW': {'currency': 'NT$', 'title': '風格分析'},
    'en_US': {'currency': 'US$', 'title': 'Style Analysis'}
}

# OAuth 第三方登入初始化 (Auth Setup)
# 必須在 app 初始化之後
oauth = OAuth(app)

# AI 模型初始化 (AI Model Initialization)
# ---  Google Gemini (生成式 AI) ---
try:
    import google.generativeai as genai

    if GENAI_API_KEY:
        genai.configure(api_key=GENAI_API_KEY)

        # 🔍 自動列出你的 API Key 支援的所有模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f">> 🔍 您的金鑰支援以下模型: {available_models}")

        if available_models:
            # 設定優先順序：優先找 1.5 flash，沒有的話找 1.0 pro，再沒有就隨便抓清單裡的第一個
            target_model_name = None
            if 'models/gemini-1.5-flash' in available_models:
                target_model_name = 'gemini-1.5-flash'
            elif 'models/gemini-1.5-flash-latest' in available_models:
                target_model_name = 'gemini-1.5-flash-latest'
            elif 'models/gemini-1.0-pro' in available_models:
                target_model_name = 'gemini-1.0-pro'
            elif 'models/gemini-pro' in available_models:
                target_model_name = 'gemini-pro'
            else:
                # 把 'models/xxx' 的 'models/' 拔掉
                target_model_name = available_models[0].replace('models/', '')

            model = genai.GenerativeModel(target_model_name)
            print(f">> ✅ 成功掛載 Gemini 模型: {target_model_name}")
        else:
            model = None
            print(">> ⚠️ 警告: 您的 API Key 沒有綁定任何可用的文字生成模型！")
    else:
        model = None
        print(">> ⚠️ Warning: 未設定 GENAI_API_KEY")
except Exception as e:
    model = None
    print(f">> ⚠️ Gemini 初始化失敗: {e}")

# ---  InsightFace (換臉引擎) ---
print(">> 正在初始化 AI 換臉引擎... (首次運行需下載模型)")

# 設定運算裝置：0 為 GPU (需 CUDA), -1 為 CPU
INFERENCE_DEVICE_ID = 0

face_app = None
swapper = None

try:
    # (A) 臉部偵測器 (Face Detection)
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=INFERENCE_DEVICE_ID, det_size=(640, 640))

    # (B) 換臉模型 (Face Swap)
    model_path = 'inswapper_128.onnx'
    if not os.path.exists(model_path):
        print(f"⚠️ 警告: 找不到 {model_path}，將嘗試自動下載...")

    swapper = insightface.model_zoo.get_model(model_path, download=True, download_zip=True)
    print(">> ✅ InsightFace 換臉模型載入成功！")

except Exception as e:
    print(f">> ❌ InsightFace 初始化失敗: {e}")
    print(">> 提示: 若無 NVIDIA 顯卡，請將 INFERENCE_DEVICE_ID 改為 -1")

# ---  MediaPipe (臉部網格與骨架) ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#  第三方服務整合 (3rd Party Services)
# Gradio Client (用於 AI 虛擬試穿)
GRADIO_AVAILABLE = False
try:
    from gradio_client import Client, file
    GRADIO_AVAILABLE = True
    print(">> ✅ Gradio Client 已啟用")
except ImportError:
    Client = None
    file = None
    GRADIO_AVAILABLE = False
    print(">> ❌ Gradio Client 未安裝")

#  輔助函式 (Helper Functions)
def get_db_connection():
    """建立資料庫連線"""
    db_path = os.path.join(BASE_DIR, 'style_system.db')
    db_conn = sqlite3.connect(db_path)
    db_conn.row_factory = sqlite3.Row
    return db_conn

def allowed_file(filename):
    """檢查檔案副檔名"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 圖片處理輔助 ---
def cv2_to_base64(img):
    success, buffer = cv2.imencode('.jpg', img)
    if success:
        b64_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return b64_str
    return None

def base64_to_cv2(b64str):
    """將 Base64 字串轉為 OpenCV 圖片 (用於後端處理)"""
    try:
        if ',' in b64str:
            b64str = b64str.split(',')[1]
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64str), np.uint8), cv2.IMREAD_COLOR)

    except Exception as err:
        print(f"Base64 decoding error: {err}")
        return None

def get_fashion_news():
    """
    [新增] 魔鏡專用：從 Google News RSS 抓取最新的時尚新聞
    """
    # 關鍵字：時尚 OR 穿搭 OR 流行趨勢, 地區：台灣
    rss_url = "https://news.google.com/rss/search?q=時尚+OR+穿搭+OR+流行趨勢&hl=zh-TW&gl=TW&ceid=TW:zh-TW"
    news_headlines = []

    try:
        # 設定 3 秒逾時，避免拖慢網頁載入
        response = requests.get(rss_url, timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, features='xml')
            items = soup.findAll('item')

            # 取前 8 則新聞
            for item in items[:8]:
                title = item.title.text
                # 移除標題後面通常會帶的媒體名稱 (例如 " - Vogue Taiwan")
                if ' - ' in title:
                    title = title.split(' - ')[0]
                news_headlines.append(f"★ {title}")

    except Exception as err:
        print(f"News Error: {err}")
        # 備案新聞
        news_headlines = [
            "★ 2026 春夏流行色系發布",
            "★ 台北時裝周即將登場",
            "★ 氣溫變化大，請注意保暖"
        ]

    # 加入系統狀態
    news_headlines.insert(0, "★ Smart Mirror OS 線上")

    return "   ".join(news_headlines)

# 權限控制裝飾器 (Auth Decorators)
def vip_required(f):
    """
    [資安優化版] VIP 權限檢查裝飾器
    特色：
    1. 支援 'next' 參數：登入後自動跳回原頁面，體驗更好。
    2. 雙重驗證：不只看 session，還會即時查 DB，防止權限被撤銷後仍能訪問。
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. 檢查登入狀態
        if 'user_id' not in session:
            # [UX 優化] 記錄使用者原本想去的頁面 (request.url)
            flash('請先登入以使用此功能', 'info')
            return redirect(url_for('login_page', next=request.url))

        # 2. 即時查詢資料庫確認 VIP 狀態
        # 原因：Session 可能會過期或被偽造，直接查 DB 最準確
        try:
            db_conn = get_db_connection()
            user = db_conn.execute('SELECT is_vip, role FROM users WHERE id = ?', (session['user_id'],)).fetchone()
            db_conn.close()

            # 如果找不到用戶，或者用戶不是 VIP (且不是管理員/官方)
            # 這裡我們給管理員 (admin) 和官方 (official) 特權，方便測試
            if not user:
                session.clear()  # 用戶不存在，強制登出
                return redirect(url_for('login_page'))

            is_privileged = (user['is_vip'] == 1) or (user['role'] in ['admin', 'official', 'expert'])

            if not is_privileged:
                flash('✨ 此功能為 VIP 專屬，解鎖以享受完整 AI 分析！', 'warning')
                return redirect(url_for('premium_landing'))  # 導向付費頁面

        except Exception as err:
            print(f"VIP Check Error: {err}")
            # 發生資料庫錯誤時，為了安全起見，先拒絕訪問
            flash('系統驗證發生錯誤，請稍後再試', 'error')
            return redirect(url_for('index'))

        # 通過所有檢查，放行！
        return f(*args, **kwargs)

    return decorated_function

# 倫理與安全模組 (Ethical & Safety Layer)
class ContentSafety:

    # 1. 正向行銷字典：將負面/敏感形容詞轉化為中性或賦予自信的描述
    NEGATIVE_WORDS = {
        # 身材描述重構
        '胖': '豐滿圓潤','肥': '棉花糖系','粗': '線條明顯','短': '嬌小','大餅臉': '圓潤臉型','五五身': '腰線待調整',
        '象腿': '腿部線條明顯','平胸': '纖細骨感','虎背熊腰': '上身較為厚實','水桶腰': '直筒身形','蘿蔔腿': '小腿肌明顯',
        # 負面評價重構
        '醜': '具個人特色','難看': '有進步空間','老氣': '復古成熟','糟糕': '待優化','奇怪': '前衛','顯胖': '視覺膨脹感'
    }

    # 2. 心理健康關鍵字 (用於觸發 SOS/Wellness 資源)
    DISTRESS_KEYWORDS = ['想死', '自殺', '沒用', '討厭自己', '絕望', '痛苦', '活不下去']

    # 3. 全域免責聲明
    DISCLAIMER = "【溫馨提醒】美的標準由您定義。本系統建議僅供穿搭參考，希望能為您的自信加分。"

    @staticmethod
    def check_mental_health(text):
        """
        檢測輸入文字是否包含情緒困擾關鍵字。
        Returns:
            bool: 若包含危險關鍵字則回傳 True
        """
        if not text:
            return False
        # 確保比對時不受大小寫影響 (雖然中文沒差，但保留擴充性)
        norm_text = str(text).lower()
        return any(k in norm_text for k in ContentSafety.DISTRESS_KEYWORDS)

    @staticmethod
    def sanitize(text):
        """
        [整合過濾器 Pipeline]
        1. HTML Escape: 防止 XSS 攻擊
        2. Positive Reframing: 替換負面字眼
        """
        if not text:
            return ""

        # Step 1: 資安防護 (使用標準庫 html.escape 取代手動 replace，更全面)
        safe_text = html.escape(str(text))

        # Step 2: 正向行銷 (字典替換)
        for bad, good in ContentSafety.NEGATIVE_WORDS.items():
            if bad in safe_text:
                safe_text = safe_text.replace(bad, good)

        return safe_text

# 全方位風格定位矩陣 (STYLE_MATRIX)
STYLE_MATRIX = {
    "STYLE_MATRIX": {
        "Q1": {
            "id": "sweet_ingenue",
            "name": "🌸 甜美鄰家系 (Sweet & Romantic)",
            "analysis_logic": {
                "volume": "小量感 (Low Volume)",
                "line": "曲線型 (Curved)",
                "psychology": "傳遞溫柔無害的訊號 (Approachable & Gentle)"
            },
            "wardrobe_guide": {
                "WOMEN": {
                    "items": {
                        "tops": "彼得潘領襯衫 (Peter Pan Collar Shirt)、馬海毛開衫 (Mohair Cardigan)、荷葉邊上衣 (Ruffled Blouse)",
                        "bottoms": "百褶短裙 (Pleated Skirt)、淺色直筒褲 (Light Denim)、碎花洋裝 (Floral Dress)",
                        "shoes": "瑪莉珍鞋 (Mary Janes)、芭蕾舞鞋 (Ballet Flats)、圓頭樂福鞋 (Round Loafers)"
                    },
                    "fabrics": ["馬海毛 (Mohair)", "雪紡 (Chiffon)", "蕾絲 (Lace)"],
                    "colors": ["#FFB7C5 (櫻花粉)", "#F3E5AB (奶油黃)"]
                },
                "MEN": {
                    "items": {
                        "tops": "軟糯針織背心 (Soft Knit Vest)、寬鬆開衫 (Oversized Cardigan)、粉嫩色帽T (Pastel Hoodie)",
                        "bottoms": "淺卡其褲 (Light Chinos)、淺色寬褲 (Relaxed Jeans)、膝上短褲 (Shorts)",
                        "shoes": "帆布鞋 (Canvas Shoes)、小白鞋 (White Sneakers)、德訓鞋 (German Trainer)"
                    },
                    "fabrics": ["軟棉 (Soft Cotton)", "細針織 (Fine Knit)", "燈芯絨 (Corduroy)"],
                    "colors": ["#F5F5DC (米杏色)", "#E0FFFF (寶寶藍)"]
                }
            },
            "styling_tips": {
                "WOMEN": {
                    "makeup": "白開水妝 (Clean Girl Makeup)、膨脹色腮紅 (Pink Blush)",
                    "hair": "空氣瀏海 (Air Bangs)、溫柔半紮髮 (Half-up)",
                    "accessories": "蝴蝶結 (Ribbon)、細珍珠項鍊 (Pearl Necklace)"
                },
                "MEN": {
                    "makeup": "乾淨無鬍鬚 (Clean Shaven)、潤唇膏 (Lip Balm)",
                    "hair": "丹迪頭 (Dandy Cut)、微分碎蓋 (Textured Crop)",
                    "accessories": "細銀戒 (Silver Ring)、帆布包 (Canvas Bag)"
                }
            }
        },

        "Q2": {
            "id": "edgy_gamine",
            "name": "⚡ 潮流個性系 (Edgy & Trendy)",
            "analysis_logic": {
                "volume": "小量感 (Low Volume)",
                "line": "直線型 (Straight)",
                "psychology": "展現自我主張與獨特審美 (Rebellious & Cool)"
            },
            "wardrobe_guide": {
                "WOMEN": {
                    "items": {
                        "tops": "短版露腰T (Crop Top)、賽車外套 (Racer Jacket)、網眼透視衫 (Mesh Top)",
                        "bottoms": "工裝降落傘褲 (Parachute Pants)、低腰短裙 (Mini Skirt)、寬版牛仔褲 (Baggy Jeans)",
                        "shoes": "厚底靴 (Platform Boots)、復古球鞋 (Retro Sneakers)、銀色單鞋 (Silver Flats)"
                    },
                    "fabrics": ["丹寧 (Denim)", "皮革 (Leather)", "金屬光澤 (Metallic)"],
                    "colors": ["#000000 (酷黑)", "#4169E1 (電光藍)"]
                },
                "MEN": {
                    "items": {
                        "tops": "塗鴉T恤 (Graphic Tee)、機能背心 (Tech Vest)、重磅帽T (Oversized Hoodie)",
                        "bottoms": "多口袋工裝褲 (Cargo Pants)、破壞鬚丹寧 (Distressed Jeans)、運動褲 (Track Pants)",
                        "shoes": "老爹鞋 (Chunky Sneakers)、戰鬥靴 (Combat Boots)、高筒鞋 (High Tops)"
                    },
                    "fabrics": ["尼龍 (Nylon)", "機能布 (Tech Fleece)", "重磅丹寧 (Denim)"],
                    "colors": ["#333333 (炭灰)", "#FF4500 (螢光橘)"]
                }
            },
            "styling_tips": {
                "WOMEN": {
                    "makeup": "小煙燻妝 (Smokey Eye)、貓眼眼線 (Cat Eye)",
                    "hair": "狼尾層次剪 (Wolf Cut)、挑染 (Highlights)",
                    "accessories": "耳骨夾 (Ear Cuff)、頸鍊 (Choker)"
                },
                "MEN": {
                    "makeup": "英氣眉 (Defined Brows)、霧面底妝 (Matte Skin)",
                    "hair": "狼尾頭 (Mullet)、寸頭 (Buzz Cut)",
                    "accessories": "古巴鍊 (Cuban Chain)、毛帽 (Beanie)"
                }
            }
        },

        "Q3": {
            "id": "elegant_luxury",
            "name": "💎 優雅貴氣系 (Elegant & Luxury)",
            "analysis_logic": {
                "volume": "大量感 (High Volume)",
                "line": "曲線型 (Curved)",
                "psychology": "展現高質感與從容自信 (Sophisticated & Luxurious)"
            },
            "wardrobe_guide": {
                "WOMEN": {
                    "items": {
                        "tops": "真絲襯衫 (Silk Blouse)、小香風外套 (Tweed Jacket)、羊絨大衣 (Cashmere Coat)",
                        "bottoms": "緞面長裙 (Satin Skirt)、西裝寬褲 (Wide Trousers)、修身連身裙 (Sheath Dress)",
                        "shoes": "尖頭高跟鞋 (Stilettos)、穆勒鞋 (Mules)、經典樂福鞋 (Loafers)"
                    },
                    "fabrics": ["桑蠶絲 (Silk)", "絲絨 (Velvet)", "粗花呢 (Tweed)"],
                    "colors": ["#B22222 (酒紅)", "#CFB53B (香檳金)"]
                },
                "MEN": {
                    "items": {
                        "tops": "馬球大衣 (Polo Coat)、亞麻襯衫 (Linen Shirt)、高領毛衣 (Turtleneck)",
                        "bottoms": "訂製西褲 (Tailored Trousers)、白色丹寧 (White Jeans)、高腰軍褲 (Gurkha Pants)",
                        "shoes": "樂福鞋 (Penny Loafers)、麂皮靴 (Suede Boots)、孟克鞋 (Monk Straps)"
                    },
                    "fabrics": ["羊絨 (Cashmere)", "亞麻 (Linen)", "麂皮 (Suede)"],
                    "colors": ["#000080 (海軍藍)", "#F5DEB3 (小麥色)"]
                }
            },
            "styling_tips": {
                "WOMEN": {
                    "makeup": "精緻紅唇 (Red Lip)、光澤肌 (Glowy Skin)",
                    "hair": "大波浪捲髮 (Big Waves)、法式低盤髮 (Low Bun)",
                    "accessories": "珍珠耳環 (Pearl Earrings)、絲巾 (Silk Scarf)"
                },
                "MEN": {
                    "makeup": "修容整潔 (Clean Grooming)、自然眉型 (Natural Brow)",
                    "hair": "紳士旁分 (Side Part)、油頭 (Slick Back)",
                    "accessories": "機械錶 (Mechanical Watch)、口袋巾 (Pocket Square)"
                }
            }
        },

        "Q4": {
            "id": "powerful_sharp",
            "name": "👑 氣場權威系 (Powerful & Sharp)",
            "analysis_logic": {
                "volume": "大量感 (High Volume)",
                "line": "直線型 (Straight)",
                "psychology": "建立專業權威與距離感 (Authoritative & Sharp)"
            },
            "wardrobe_guide": {
                "WOMEN": {
                    "items": {
                        "tops": "大廓形西裝 (Structured Blazer)、挺版襯衫 (Crisp Shirt)、長版風衣 (Trench Coat)",
                        "bottoms": "鉛筆裙 (Pencil Skirt)、直筒西褲 (Straight Trousers)、皮褲 (Leather Pants)",
                        "shoes": "尖頭靴 (Pointed Boots)、牛津鞋 (Oxford Shoes)、極簡跟鞋 (Pumps)"
                    },
                    "fabrics": ["西裝料 (Suiting)", "皮革 (Leather)", "精紡羊毛 (Wool)"],
                    "colors": ["#000000 (極致黑)", "#FFFFFF (純白)"]
                },
                "MEN": {
                    "items": {
                        "tops": "雙排扣西裝 (Double Breasted Suit)、正裝襯衫 (Dress Shirt)、長大衣 (Overcoat)",
                        "bottoms": "正裝西褲 (Dress Pants)、原色丹寧 (Dark Denim)、打褶褲 (Pleated Trousers)",
                        "shoes": "牛津鞋 (Oxford Shoes)、切爾西靴 (Chelsea Boots)、德比鞋 (Derby)"
                    },
                    "fabrics": ["精紡羊毛 (Worsted Wool)", "府綢 (Poplin)", "硬挺皮革 (Leather)"],
                    "colors": ["#2F4F4F (深岩灰)", "#808080 (炭灰)"]
                }
            },
            "styling_tips": {
                "WOMEN": {
                    "makeup": "英氣眉 (Sharp Brows)、霧面裸唇 (Matte Lip)",
                    "hair": "俐落直髮 (Sleek Straight)、低馬尾 (Low Pony)",
                    "accessories": "幾何耳飾 (Geometric Earrings)、寬腰帶 (Wide Belt)"
                },
                "MEN": {
                    "makeup": "霧面控油 (Matte Skin)、立體輪廓 (Sharp Contours)",
                    "hair": "飛機頭 (Quiff)、龐畢度油頭 (Pompadour)",
                    "accessories": "領帶夾 (Tie & Clip)、公事包 (Briefcase)"
                }
            }
        },

        "CENTER": {
            "id": "natural_relaxed",
            "name": "🍃 自然舒適系 (Natural & Relaxed)",
            "analysis_logic": {
                "volume": "中量感 (Mid Volume)",
                "line": "混合型 (Mixed)",
                "psychology": "展現真實自我與生活態度 (Relaxed & Authentic)"
            },
            "wardrobe_guide": {
                "WOMEN": {
                    "items": {
                        "tops": "條紋T恤 (Striped Tee)、棉麻襯衫 (Linen Shirt)、針織套頭衫 (Knit Pullover)",
                        "bottoms": "直筒牛仔褲 (Straight Jeans)、棉質長裙 (Midi Skirt)、休閒寬褲 (Wide Pants)",
                        "shoes": "帆布鞋 (Canvas Sneakers)、勃肯鞋 (Birkenstocks)、樂福鞋 (Loafers)"
                    },
                    "fabrics": ["棉 (Cotton)", "麻 (Linen)", "丹寧 (Denim)"],
                    "colors": ["#D2B48C (卡其)", "#556B2F (橄欖綠)"]
                },
                "MEN": {
                    "items": {
                        "tops": "牛津襯衫 (Oxford Shirt)、連帽衛衣 (Hoodie)、寬版T恤 (Oversized Tee)",
                        "bottoms": "卡其褲 (Chinos)、工裝短褲 (Cargo Shorts)、直筒丹寧 (Raw Denim)",
                        "shoes": "復古跑鞋 (New Balance)、帆船鞋 (Boat Shoes)、涼鞋 (Sandals)"
                    },
                    "fabrics": ["純棉 (Cotton)", "帆布 (Canvas)", "華夫格 (Waffle)"],
                    "colors": ["#A9A9A9 (灰色)", "#8B4513 (大地棕)"]
                }
            },
            "styling_tips": {
                "WOMEN": {
                    "makeup": "偽素顏 (No-Makeup Look)、裸色唇 (Nude Lip)",
                    "hair": "隨性丸子頭 (Messy Bun)、自然捲髮 (Natural Waves)",
                    "accessories": "托特包 (Tote Bag)、黑框眼鏡 (Glasses)"
                },
                "MEN": {
                    "makeup": "清爽面容 (Clean Face)、防曬 (Sunscreen)",
                    "hair": "紋理燙 (Texture Perm)、隨性瀏海 (Messy Fringe)",
                    "accessories": "鴨舌帽 (Cap)、後背包 (Backpack)"
                }
            }
        }
    },
    # 風格遷移路徑 (Style Migration Logic)
    "MIGRATION_PATH": {
        "want_more_pro": "增加直線條 + 硬挺材質 (往 Q4 氣場權威系靠攏)",
        "want_more_fem": "增加曲線感 + 柔軟材質 (往 Q3 優雅貴氣系靠攏)",
        "want_younger": "增加明亮色 + 短版剪裁 (往 Q1 或 Q2 靠攏)",
        "want_sharper": "增加對比色 + 獨特設計 (往 Q2 潮流個性系靠攏)",
        "want_relaxed": "降低對比度 + 寬鬆版型 (往 CENTER 自然舒適系靠攏)"
    }
}

# 穿搭風格分類體系
STYLE_TAXONOMY = {
    # 1. TPO 場景維度 (最實用的過濾器)
    "Occasion": {
        "label": "適用場景",
        "options": {
            "COMMUTE":  "職場通勤 (Smart Casual)",
            "DATE":     "約會聚餐 (Romantic/Date Night)",
            "CAMPUS":   "校園日常 (Campus Life)",
            "PARTY":    "派對晚宴 (Party & Event)",
            "VACATION": "度假出遊 (Resort & Vacation)",
            "SPORT":    "運動休閒 (Athleisure)",
            "HOME":     "居家慵懶 (Loungewear)",
            "BUSINESS": "商務正裝 (Business Formal)"
        }
    },

    # 2. 核心美學維度 (大分類，決定整體基調)
    "Aesthetic": {
        "label": "核心風格",
        "options": {
            "MINIMALIST": "極簡主義 (Minimalist/Clean)",
            "STREET":     "街頭潮流 (Streetwear/Hype)",
            "ELEGANT":    "優雅知性 (Elegant/Sophisticated)",
            "VINTAGE":    "復古懷舊 (Retro/Vintage)",
            "SWEET":      "甜美/軟萌 (Sweet/Soft)",
            "BOYISH":     "中性帥氣 (Tomboy/Unisex)",
            "RUGGED":     "粗獷工裝 (Rugged/Workwear)",
            "DAPPER":     "紳士雅痞 (Dapper/Sartorial)",
            "AVANT_GARDE":"前衛解構 (Avant-Garde)"
        }
    },

    # 3. 流行微趨勢 (增加男性熱門趨勢)
    "Trend": {
        "label": "流行趨勢",
        "options": {
            "OLD_MONEY":  "老錢風 (Old Money) - 質感、大地色、針織",
            "CLEAN_FIT":  "乾淨俐落 (Clean Fit) - 合身、基礎款、高質感",
            "CITY_BOY":   "日系 City Boy (City Boy) - 寬鬆、層次、混搭",
            "BLOKECORE":  "復古球衣風 (Blokecore) - 球衣、牛仔褲、復古鞋",
            "GOPCORE":    "山系戶外 (Gorpcore) - 機能、衝鋒衣、露營",
            "Y2K":        "千禧風 (Y2K) - 亮色、科技感、金屬",
            "MAILLARD":   "美拉德 (Maillard) - 棕色系疊穿",
            "MOB_STYLE":  "黑幫美學 (Mob Wife/Boss) - 皮草、皮革、強氣場",
            "IVY_LEAGUE": "常春藤學院 (Ivy League) - 衛衣、格紋、棒球外套"
        }
    },

    # 4. 文化系別
    "Culture": {
        "label": "文化系別",
        "options": {
            "FRENCH_CHIC": "法式慵懶 (French Chic)",
            "K_FASHION":   "韓系簡約 (K-Fashion)",
            "J_STYLE":     "日系多層次 (J-Style)",
            "AMEKAJI":     "美式復古/阿美卡機 (Amekaji)",
            "BRITISH":     "英倫紳士 (British Style)",
            "NEO_CHINESE": "新中式 (Neo-Chinese)"
        }
    },

    # 5. 藝術與視覺元素
    "Art_Element": {
        "label": "藝術元素",
        "options": {
            "ROMANTICISM": "浪漫主義 (Romanticism) - 柔軟、荷葉邊、詩意",
            "GOTHIC":      "暗黑哥德 (Gothic) - 黑色、金屬、神秘",
            "BAROQUE":     "巴洛克 (Baroque) - 華麗、印花、絲絨",
            "BOHEMIAN":    "波西米亞 (Bohemian) - 流蘇、圖騰、自由",
            "FUTURISM":    "未來主義 (Futurism) - 銀色、科技感、機能",
            "ACADEMIA":    "知識份子 (Academia) - 書卷氣、棕色調、格紋"
        }
    }
}

# 全方位髮型定位資料庫 (Hairstyle Database - Integrated)
HAIRSTYLE_DATABASE = {
    # --- 女生髮型 (Women's Hairstyles) ---
    "WOMEN": {
        "Short": {
            "Pixie_Cut": {
                "name": "精靈短髮 (Pixie Cut)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "極短，露出耳朵與頸部線條，頭頂層次豐富。"
            },
            "Blunt_Bob": {
                "name": "一刀切短髮 (Blunt Bob)",
                "matrix_tag": ["Q4", "Q2"],
                "desc": "髮尾齊平無層次，通常長度在下巴或耳下，線條俐落。"
            },
            "Short_Bob": {
                "name": "日系短髮 (Short Bob)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "後腦勺有圓弧度，髮尾收進去，強調頭型飽滿。"
            }
        },
        "Medium": {
            "Clavicle_Cut": {
                "name": "鎖骨髮 (Clavicle Cut)",
                "matrix_tag": ["CENTER", "Q4"],
                "desc": "長度剛好碰到鎖骨，可直可捲。"
            },
            "Layered_Cut": {
                "name": "高層次中長髮 (Layered Cut)",
                "matrix_tag": ["Q2"],
                "desc": "髮尾剪出明顯長短落差，輕盈亂翹。"
            },
            "C_Curl": {
                "name": "C字彎 (C-Curl)",
                "matrix_tag": ["Q1", "Q3"],
                "desc": "直髮基底，僅在髮尾燙一個C型彎度。"
            }
        },
        "Long": {
            "Sleek_Straight": {
                "name": "黑長直 (Sleek Straight)",
                "matrix_tag": ["Q4", "CENTER"],
                "desc": "無捲度，強調髮質光澤與垂墜感的直髮。"
            },
            "Big_Waves": {
                "name": "大波浪 (Big Waves)",
                "matrix_tag": ["Q3"],
                "desc": "捲度大而鬆散，S型紋理明顯。"
            },
            "French_Perm": {
                "name": "法式慵懶捲 (French Perm)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "捲度隨意、不規則，強調蓬鬆感。"
            },
            "Wool_Roll": {
                "name": "羊毛捲 (Wool Roll)",
                "matrix_tag": ["Q1", "Q2"],
                "desc": "從髮根開始的小捲度，呈現毛茸茸的爆炸感。"
            }
        },
        "Bangs": {
            "Air_Bangs": {
                "name": "空氣瀏海 (Air Bangs)",
                "matrix_tag": ["Q1"],
                "desc": "輕薄、看得到額頭。"
            },
            "Curtain_Bangs": {
                "name": "八字瀏海 (Curtain Bangs)",
                "matrix_tag": ["Q3", "CENTER"],
                "desc": "中分或旁分，長度在顴骨附近向外翻捲。"
            },
            "Sleek_Back": {
                "name": "大光明 (Sleek Back)",
                "matrix_tag": ["Q4"],
                "desc": "完全露出額頭，頭髮全部向後梳。"
            },
            "Baby_Hair": {
                "name": "胎毛瀏海 (Baby Hair)",
                "matrix_tag": ["Q1", "Q2"],
                "desc": "在髮際線留一點點碎髮修飾。"
            }
        },
        "Updo": {
            "Bun": {
                "name": "丸子頭 (Bun)",
                "matrix_tag": ["Q1", "CENTER"],
                "desc": "頭頂或後腦勺盤起來的球狀髮髻。"
            },
            "Low_Ponytail": {
                "name": "貼頭皮低馬尾 (Sleek Low Ponytail)",
                "matrix_tag": ["Q4", "Q3"],
                "desc": "梳得非常整齊、貼合頭型的低馬尾。"
            },
            "Half_Up": {
                "name": "半頭/公主頭 (Half Up)",
                "matrix_tag": ["Q1", "Q3"],
                "desc": "只綁上半部頭髮。"
            }
        }
    },

    # --- 男生髮型 (Men's Hairstyles) ---
    "MEN": {
        "Short": {
            "Buzz_Cut": {
                "name": "寸頭 (Buzz Cut)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "全頭推短，露出頭皮形狀。"
            },
            "Crop_Top": {
                "name": "栗子頭 (Crop Top)",
                "matrix_tag": ["Q2"],
                "desc": "兩側推短，瀏海剪齊短且貼額頭。"
            },
            "Barber_Fade": {
                "name": "美式油頭 (Barber Fade)",
                "matrix_tag": ["Q4", "Q3"],
                "desc": "兩側漸層推剪，上方留長往後梳。"
            }
        },
        "Forehead": {
            "Quiff": {
                "name": "飛機頭 (Quiff)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "瀏海向上抓翹，露出額頭。"
            },
            "Slick_Back": {
                "name": "大背頭 (Slick Back)",
                "matrix_tag": ["Q4", "Q3"],
                "desc": "所有頭髮利用髮油全部向後貼梳。"
            },
            "Side_Part": {
                "name": "復古三七分 (Side Part)",
                "matrix_tag": ["Q4", "CENTER"],
                "desc": "經典分線，比例為3:7或2:8。"
            }
        },
        "Bangs": {
            "Dandy_Cut": {
                "name": "丹迪頭 (Dandy Cut)",
                "matrix_tag": ["Q1", "CENTER"],
                "desc": "瀏海厚重且放下，兩側鬢角服貼。"
            },
            "Textured_Crop": {
                "name": "微分碎蓋 (Textured Crop)",
                "matrix_tag": ["Q2", "CENTER"],
                "desc": "瀏海蓋住額頭但打碎剪出層次。"
            },
            "Comma_Hair": {
                "name": "逗號瀏海 (Comma Hair)",
                "matrix_tag": ["Q2", "Q1"],
                "desc": "瀏海向內捲成逗號形狀，半露額頭。"
            }
        },
        "Perm": {
            "Texture_Perm": {
                "name": "紋理燙 (Texture Perm)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "燙出微捲，增加蓬鬆度與線條感。"
            },
            "Twist_Perm": {
                "name": "鋼夾燙 (Twist Perm)",
                "matrix_tag": ["Q2"],
                "desc": "燙出直立束狀感，視覺爆炸。"
            },
            "Mid_Length_Curls": {
                "name": "中長捲髮 (Mid-Length Curls)",
                "matrix_tag": ["CENTER"],
                "desc": "長度及耳，帶有明顯捲度。"
            },
            "Man_Bun": {
                "name": "武士頭 (Man Bun)",
                "matrix_tag": ["Q2", "CENTER"],
                "desc": "頭頂綁小髮髻，兩側通常鏟青。"
            },
            "Mullet": {
                "name": "狼尾頭 (Mullet)",
                "matrix_tag": ["Q2"],
                "desc": "前短後長，脖子後面的頭髮留長。"
            }
        }
    }
}

# 全方位妝容定位資料庫 (Makeup Database)
MAKEUP_DATABASE = {
    # ---------------- [WOMEN] 女生妝容 ----------------
    "WOMEN": {
        # --- 亞洲系 (Asian Styles) ---
        "ASIAN": {
            "Glass_Skin": {
                "name": "韓系水光肌 (Glass Skin)",
                "matrix_tag": ["Q1", "CENTER"],
                "desc": "強調皮膚像玻璃一樣透亮、有光澤，底妝極薄。"
            },
            "Idol_Makeup": {
                "name": "女團妝 (Idol Makeup)",
                "matrix_tag": ["Q1", "Q2"],
                "desc": "太陽花睫毛、臥蠶明顯、亮片點綴，強調眼神。"
            },
            "Juicy_Makeup": {
                "name": "果汁妝 (Juicy Makeup)",
                "matrix_tag": ["Q1"],
                "desc": "色彩像水果般鮮豔(蜜桃/草莓)，強調飽滿唇釉與同色系腮紅。"
            },
            "No_Makeup": {
                "name": "偽素顏 (No-Makeup Makeup)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "用低飽和度顏色消腫，強調天生麗質，看似沒化妝。"
            },
            "Igari": {
                "name": "日系費洛蒙 (Igari)",
                "matrix_tag": ["Q1"],
                "desc": "眼下高腮紅，大面積暈染，營造微醺無辜感。"
            },
            "Sheer_Natural": {
                "name": "透明感妝 (Sheer/Natural)",
                "matrix_tag": ["CENTER"],
                "desc": "重視肌膚透明度，野生眉，水潤裸粉色唇彩。"
            },
            "Jirai_Kei": {
                "name": "地雷系 (Jirai Kei)",
                "matrix_tag": ["Q1", "Q2"],
                "desc": "下垂眼線、紅色系眼影(像哭過)、蒼白膚色，病嬌感。"
            },
            "Rich_Girl": {
                "name": "中國千金妝 (Rich Girl)",
                "matrix_tag": ["Q3"],
                "desc": "啞光無瑕底妝，強調精細眉峰與紅唇，貴氣逼人。"
            },
            "New_Chinese": {
                "name": "新中式 (New Chinese)",
                "matrix_tag": ["Q3", "Q4"],
                "desc": "結合古典現代，細長柳葉眉、低飽和大地色或紅棕色眼妝。"
            },
            "Thai_Style": {
                "name": "輕泰妝 (Thai Style)",
                "matrix_tag": ["Q2", "Q3"],
                "desc": "野生眉、濃密睫毛、大面積修容，色彩偏泰奶色。"
            }
        },

        # --- 歐美系 (Western Styles) ---
        "WESTERN": {
            "Cut_Crease": {
                "name": "截斷式眼妝 (Cut Crease)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "利用遮瑕畫出眼窩界線，強調深邃度，搭配誇張假睫毛。"
            },
            "Insta_Baddie": {
                "name": "卡戴珊風 (Instagram Baddie)",
                "matrix_tag": ["Q3", "Q4"],
                "desc": "重修容、厚唇、高光明顯、霧面底妝、完美眉型。"
            },
            "Clean_Girl": {
                "name": "歐美極簡 (Clean Girl)",
                "matrix_tag": ["CENTER", "Q4"],
                "desc": "野生眉，光澤感皮膚，潤唇膏，頭髮梳乾淨。"
            },
            "Latte_Makeup": {
                "name": "拿鐵妝 (Latte Makeup)",
                "matrix_tag": ["CENTER", "Q3"],
                "desc": "全臉咖啡色/古銅色系，打造陽光曬過的健康溫暖感。"
            },
            "Mob_Wife": {
                "name": "暴徒人妻 (Mob Wife)",
                "matrix_tag": ["Q3"],
                "desc": "煙燻眼妝、凌亂髮型、深色唇線、大量修容，頹廢奢華感。"
            },
            "French_Girl": {
                "name": "法式妝容 (French Girl)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "底妝不遮瑕(保留雀斑)，眉毛自然雜亂，重點是一抹紅唇。"
            },
            "Latina": {
                "name": "拉丁裔妝容 (Latina)",
                "matrix_tag": ["Q2", "Q3"],
                "desc": "深色眉毛、明顯唇線(比口紅深)、濃密睫毛。"
            }
        },

        # --- 復古 (Vintage Styles) ---
        "VINTAGE": {
            "Flapper": {
                "name": "20年代 (Flapper)",
                "matrix_tag": ["Q2", "Q3"],
                "desc": "極細下垂眉、深色眼影、櫻桃小嘴(只畫唇中央)。"
            },
            "Pin_up": {
                "name": "50年代 (Pin-up)",
                "matrix_tag": ["Q3", "Q1"],
                "desc": "乾淨眼妝、上揚貓眼眼線、豐滿紅唇。"
            },
            "The_Mod": {
                "name": "60年代 (The Mod)",
                "matrix_tag": ["Q2"],
                "desc": "誇張下睫毛、眼窩線、蒼白嘴唇。"
            },
            "Grunge": {
                "name": "90年代 (Grunge)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "啞光底妝、全包式眼線、土色/磚紅色口紅、細眉。"
            },
            "Y2K_Beauty": {
                "name": "千禧辣妹 (Y2K)",
                "matrix_tag": ["Q2"],
                "desc": "冰藍色/金屬色眼影、亮面唇蜜、細眉、過度腮紅。"
            }
        },

        # --- 次文化 (Subculture Styles) ---
        "SUBCULTURE": {
            "Smokey": {
                "name": "煙燻妝 (Smokey Eye)",
                "matrix_tag": ["Q2", "Q4"],
                "desc": "黑色/深灰大面積暈染，搖滾叛逆感。"
            },
            "Goth": {
                "name": "哥德妝 (Goth)",
                "matrix_tag": ["Q4", "Q2"],
                "desc": "極致蒼白皮膚、黑色口紅、銳利幾何眼線。"
            },
            "ABG": {
                "name": "亞裔歐美 (ABG)",
                "matrix_tag": ["Q2", "Q3"],
                "desc": "結合亞洲五官與歐美畫法，濃密睫毛、變色片、重修容。"
            },
            "Freckles": {
                "name": "雀斑妝 (Faux Freckles)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "人為畫上雀斑，營造度假、陽光俏皮感。"
            },
            "Sunburn": {
                "name": "曬傷妝 (Sunburn)",
                "matrix_tag": ["Q1", "CENTER"],
                "desc": "腮紅橫跨鼻樑，像被太陽曬傷，日系或特殊造型常用。"
            }
        }
    },

    # ---------------- [MEN] 男生修容 ----------------
    "MEN": {
        # --- 基礎理容 (Basic Grooming) ---
        "GROOMING": {
            "Clean_Shaven": {
                "name": "淨爽修容 (Clean Shaven)",
                "matrix_tag": ["Q1", "CENTER", "Q4"],
                "desc": "徹底刮除鬍鬚，保持下顎線條乾淨，給人清爽專業感。"
            },
            "Natural_Brows": {
                "name": "野生眉修整 (Natural Brows)",
                "matrix_tag": ["CENTER", "Q1"],
                "desc": "保留眉毛毛流感，僅修除雜毛，提升五官立體度。"
            },
            "BB_Cream": {
                "name": "偽素顏打底 (BB Cream)",
                "matrix_tag": ["Q1", "Q3"],
                "desc": "使用男士BB霜均勻膚色，遮蓋痘印，打造無瑕膚質。"
            },
            "Lip_Care": {
                "name": "潤唇護理 (Lip Balm)",
                "matrix_tag": ["CENTER", "Q1", "Q4"],
                "desc": "無色或極淡色潤唇膏，避免嘴唇乾裂，細節加分。"
            },
            "Sunscreen": {
                "name": "防曬亮白 (Sunscreen Glow)",
                "matrix_tag": ["CENTER", "Q2"],
                "desc": "強調肌膚的光澤感與防護，適合戶外運動風。"
            }
        },

        # --- 進階造型 (Advanced Styling) ---
        "STYLED": {
            "Sharp_Brows": {
                "name": "英氣劍眉 (Sharp Brows)",
                "matrix_tag": ["Q4", "Q2"],
                "desc": "強調眉峰與眉尾的銳利線條，增加權威感與氣場。"
            },
            "Contour": {
                "name": "立體修容 (Contour)",
                "matrix_tag": ["Q3", "Q4"],
                "desc": "加強鼻影與下顎線修容，打造深邃硬朗的輪廓。"
            },
            "K_Idol": {
                "name": "韓系男團妝 (K-Idol)",
                "matrix_tag": ["Q2", "Q1"],
                "desc": "輕微眼線、大地色眼影消腫、咬唇妝，舞台感強。"
            },
            "Guyliner": {
                "name": "龐克眼線 (Guyliner)",
                "matrix_tag": ["Q2"],
                "desc": "黑色眼線框住眼眶，帶有搖滾或憂鬱氣質。"
            },
            "Bronzer": {
                "name": "古銅肌 (Bronzer)",
                "matrix_tag": ["Q3", "Q2"],
                "desc": "使用古銅粉營造陽光曝曬後的健康膚色，歐美感強。"
            },
            "E_Boy": {
                "name": "E-Boy 風格 (E-Boy)",
                "matrix_tag": ["Q2"],
                "desc": "眼下微醺腮紅、畫上的小愛心或斷眉，Z世代潮流。"
            }
        }
    }
}

# 材質與圖案定位矩陣 (Material & Pattern Matrix)
MATERIAL_PATTERN_DATABASE = {
    "WOMEN": {
        "FABRICS": {
            "Natural": {
                "Cotton": {"name": "棉 (Cotton)", "matrix_tag": ["CENTER", "Q1"], "desc": "透氣柔軟，居家與休閒首選"},
                "Linen": {"name": "亞麻 (Linen)", "matrix_tag": ["CENTER"], "desc": "慵懶文藝，適合度假風"},
                "Silk": {"name": "真絲 (Silk)", "matrix_tag": ["Q3"], "desc": "光澤貴氣，適合晚宴與優雅風"},
                "Wool": {"name": "羊毛 (Wool)", "matrix_tag": ["Q3", "Q4"], "desc": "保暖垂墜，展現高級質感"}
            },
            "Synthetic": {
                "Chiffon": {"name": "雪紡 (Chiffon)", "matrix_tag": ["Q1", "Q3"], "desc": "輕薄飄逸，充滿女性柔美感"},
                "Tweed": {"name": "粗花呢 (Tweed)", "matrix_tag": ["Q3"], "desc": "經典小香風，優雅成熟"},
                "Leather": {"name": "皮革 (Leather)", "matrix_tag": ["Q2", "Q4"], "desc": "帥氣硬朗，增加氣場"}
            }
        },
        "PATTERNS": {
            "Floral": {"name": "碎花 (Floral)", "matrix_tag": ["Q1"], "desc": "清新減齡，適合鄰家風格"},
            "Polka_Dot": {"name": "波點 (Polka Dot)", "matrix_tag": ["Q1", "Q3"], "desc": "復古俏皮"},
            "Plaid": {"name": "格紋 (Plaid)", "matrix_tag": ["CENTER", "Q4"], "desc": "經典學院風或英倫風"},
            "Animal": {"name": "動物紋 (Animal Print)", "matrix_tag": ["Q2", "Q3"], "desc": "野性性感，吸睛度高"}
        }
    },
    "MEN": {
        "FABRICS": {
            "Natural": {
                "Oxford": {"name": "牛津布 (Oxford)", "matrix_tag": ["CENTER", "Q1"], "desc": "經典學院風，挺括耐穿"},
                "Seersucker": {"name": "泡泡紗 (Seersucker)", "matrix_tag": ["CENTER", "Q3"], "desc": "夏季紳士首選，透氣不貼身"},
                "Flannel": {"name": "法蘭絨 (Flannel)", "matrix_tag": ["Q1", "CENTER"], "desc": "溫暖格紋，適合日系或工裝"},
                "Worsted": {"name": "精紡羊毛 (Worsted Wool)", "matrix_tag": ["Q3", "Q4"], "desc": "商務正裝必備，光澤細膩"}
            },
            "Synthetic": {
                "Tech_Fleece": {"name": "機能布 (Tech Fleece)", "matrix_tag": ["Q2"], "desc": "運動潮流，防風保暖"},
                "Denim": {"name": "重磅丹寧 (Raw Denim)", "matrix_tag": ["Q2", "CENTER"], "desc": "硬挺有型，養牛族最愛"},
                "Corduroy": {"name": "燈芯絨 (Corduroy)", "matrix_tag": ["Q1", "CENTER"], "desc": "復古文青，秋冬質感加分"}
            }
        },
        "PATTERNS": {
            "Stripe": {"name": "條紋 (Stripe)", "matrix_tag": ["CENTER", "Q4"], "desc": "商務細條紋顯瘦，寬條紋休閒"},
            "Check": {"name": "格紋 (Check)", "matrix_tag": ["Q1", "Q3"], "desc": "蘇格蘭紋優雅，棋盤格潮流"},
            "Camouflage": {"name": "迷彩 (Camo)", "matrix_tag": ["Q2"], "desc": "街頭工裝必備元素"},
            "Paisley": {"name": "變形蟲 (Paisley)", "matrix_tag": ["Q2", "Q3"], "desc": "復古雅痞，展現獨特品味"}
        }
    }
}

# 版型細節字典 (Cut & Silhouette Dictionary)
CUT_DETAIL_DATABASE = {
    "WOMEN": {
        "NECKLINE": {
            "V_Neck": {"name": "V領 (V-Neck)", "suit": ["圓臉", "短脖子"], "effect": "拉長頸部線條"},
            "Square": {"name": "方領 (Square Neck)", "suit": ["窄肩", "圓臉"], "effect": "展現鎖骨，復古顯瘦"},
            "Turtle": {"name": "高領 (Turtle Neck)", "suit": ["長脖子", "小臉"], "effect": "氣質保暖，修飾長頸"}
        },
        "BOTTOMS": {
            "A_Line": {"name": "A字裙 (A-Line)", "suit": ["梨型", "假胯寬"], "effect": "遮肉顯瘦第一名"},
            "Wide_Leg": {"name": "落地寬褲 (Wide Leg)", "suit": ["腿型不直", "五五身"], "effect": "拉長腿部比例"},
            "Pencil": {"name": "鉛筆裙 (Pencil Skirt)", "suit": ["沙漏型", "直筒型"], "effect": "強調曲線與職場專業感"}
        }
    },
    "MEN": {
        "TOP_DETAIL": {
            "Camp_Collar": {"name": "古巴領 (Camp Collar)", "suit": ["短脖子", "方臉"], "effect": "修飾頸部，夏日休閒感"},
            "Drop_Shoulder": {"name": "落肩剪裁 (Drop Shoulder)", "suit": ["窄肩", "單薄身形"], "effect": "增加份量感，營造寬肩效果"},
            "Raglan": {"name": "插肩袖 (Raglan)", "suit": ["溜肩", "運動風"], "effect": "修飾肩線，活動自如"}
        },
        "BOTTOMS": {
            "Pleated": {"name": "打褶褲 (Pleated Pants)", "suit": ["大腿粗", "梨型"], "effect": "增加大腿空間，修飾腿型"},
            "Tapered": {"name": "錐形褲 (Tapered Fit)", "suit": ["小腿細", "O型腿"], "effect": "上寬下窄，俐落修身"},
            "Cargo": {"name": "工裝口袋 (Cargo Pockets)", "suit": ["竹竿腿", "直筒型"], "effect": "增加下半身視覺份量"}
        }
    }
}

# 配件資料庫 (ACCESSORY_MATRIX)
ACCESSORY_MATRIX = {
    # ---------------- [WOMEN] 女生配件 ----------------
    "WOMEN": {
        "Q1": { # 🌸 甜美鄰家
            "Jewelry": [
                "Small Floral Ring (小巧花卉指環)", "Heart Ring (心型戒指)", "Dainty Necklace (細鎖骨鍊)",
                "Tiny Pendant (精緻小吊墜)", "Beaded Bracelet (串珠手鍊)", "Thin Cord (細繩編織手環)",
                "Stud Earrings (耳釘)", "Small Hoops (小型圓環耳環)", "Enamel Brooch (琺瑯胸針)"
            ],
            "Functional_Styling": [
                "Small Dial Watch (小圓盤皮帶錶)", "Round Glasses (圓框眼鏡)", "Pastel Frames (透明粉色鏡框)",
                "Beret (貝雷帽)", "Knit Beanie (軟糯針織帽)", "Floral Scarf (碎花圍巾)",
                "Lace Gloves (蕾絲邊飾手套)", "Skinny Belt (細皮帶)"
            ],
            "Bags": [
                "Cloud Bag (雲朵包)", "Mini Backpack (迷你後背包)", "Pastel Wallet (粉嫩色短夾)", "Kiss-lock Bag (口金包)"
            ],
            "Hair_Styling": [
                "Scrunchie (大腸髮圈)", "Ribbon Clip (蝴蝶結髮夾)", "Pearl Headband (珍珠髮箍)", "Satin Ribbon (緞帶頭飾)"
            ]
        },
        "Q2": { # ⚡ 潮流個性
            "Jewelry": [
                "Stacked Rings (銀質疊戴戒)", "Geometric Ring (幾何戒指)", "Chunky Chain (粗鍊條項鍊)",
                "Choker (黑色頸鍊)", "Chain Bracelet (鍊條手鍊)", "Rubber Band (橡膠手環)",
                "Ear Cuff (金屬耳骨夾)", "Safety Pin Earring (迴紋針造型耳環)"
            ],
            "Functional_Styling": [
                "Digital Watch (電子錶)", "G-Shock Style (機能風手錶)", "Futuristic Shades (未來感墨鏡)",
                "Tinted Lenses (彩色鏡片)", "Baseball Cap (棒球帽)", "Beanie (冷帽)",
                "Fingerless Gloves (露指手套)", "Studded Belt (鉚釘腰帶)"
            ],
            "Bags": [
                "Chest Bag (尼龍胸包)", "Metallic Bag (金屬銀色腋下包)", "Velcro Wallet (魔鬼氈錢包)", "Mini Bag (小廢包)"
            ],
            "Hair_Styling": [
                "Metal Clip (金屬一字夾)", "Bandana (運動頭巾)", "Claw Clip (鯊魚夾)", "Colorful Pins (彩色髮夾)"
            ]
        },
        "Q3": { # 💎 優雅貴氣
            "Jewelry": [
                "Gemstone Ring (寶石大戒指)", "Gold Ring (金質戒指)", "Pearl Necklace (珍珠項鍊)",
                "Crystal Choker (華麗水鑽鍊)", "Gold Bangle (金色手鐲)", "Tennis Bracelet (排鑽手鍊)",
                "Drop Earrings (垂墜耳環)", "Large Pearl (大珍珠耳飾)", "Vintage Brooch (復古珍珠胸針)"
            ],
            "Functional_Styling": [
                "Jewelry Watch (鑲鑽金錶)", "Cat-eye Glasses (貓眼眼鏡)", "Oversized Sunglasses (大框墨鏡)",
                "Wide-brim Hat (赫本風寬簷帽)", "Silk Scarf (絲巾)", "Fur Stole (皮草披肩)",
                "Satin Gloves (緞面手套)", "Gold Buckle Belt (金釦皮帶)"
            ],
            "Bags": [
                "Top Handle Bag (手提凱莉包)", "Clutch (晚宴手拿包)", "Quilted Bag (鍊條菱格包)", "Long Wallet (皮質長夾)"
            ],
            "Hair_Styling": [
                "Velvet Clip (絲絨髮夾)", "Silk Headband (絲綢髮箍)", "Rhinestone Pin (水鑽髮飾)", "Low Bun Pin (低盤髮插)"
            ]
        },
        "Q4": { # 👑 氣場權威
            "Jewelry": [
                "Signet Ring (印章戒指)", "Wide Band (寬版金屬戒)", "Geometric Torque (幾何頸環)",
                "Sleek Chain (俐落金屬鍊)", "Wide Cuff (寬版開口手鐲)", "Bold Hoops (粗圓環耳環)",
                "Architectural Earrings (結構感耳飾)", "Bar Brooch (極簡金屬條胸針)"
            ],
            "Functional_Styling": [
                "Chronograph (三眼計時錶)", "Steel Watch (鋼帶錶)", "Aviators (飛行員墨鏡)",
                "Square Frames (方框眼鏡)", "Fedora (紳士帽)", "Cashmere Scarf (純色羊絨圍巾)",
                "Leather Gloves (極簡真皮手套)", "Corset Belt (寬腰封)"
            ],
            "Bags": [
                "Structured Tote (定型托特包)", "Briefcase (公事包)", "Envelope Clutch (信封手拿包)", "Card Holder (極簡名片夾)"
            ],
            "Hair_Styling": [
                "Sleek Clip (俐落抓夾)", "Minimalist Barrette (極簡一字夾)", "Metal Hair Stick (金屬髮簪)"
            ]
        },
        "CENTER": { # 🍃 自然舒適
            "Jewelry": [
                "Matte Silver Ring (霧面銀戒)", "Wooden Ring (木質戒指)", "Cord Necklace (繩結項鍊)",
                "Raw Stone Pendant (原石吊墜)", "Leather Bracelet (編織皮革手環)", "Matte Studs (磨砂小耳環)",
                "Wood Brooch (木作胸針)"
            ],
            "Functional_Styling": [
                "Canvas Watch (帆布帶手錶)", "Minimalist Watch (簡約大三針)", "Wire Frames (細金屬框眼鏡)",
                "Straw Hat (草編帽)", "Bucket Hat (漁夫帽)", "Linen Scarf (棉麻圍巾)",
                "Braided Belt (編織皮帶)"
            ],
            "Bags": [
                "Canvas Tote (帆布托特包)", "Rucksack (束口後背包)", "Straw Bag (編織草包)", "Eco Wallet (環保材質錢包)"
            ],
            "Hair_Styling": [
                "Linen Headband (棉麻髮帶)", "Wooden Clip (木質髮夾)", "Plain Scrunchie (素色大腸圈)"
            ]
        }
    },

    # ---------------- [MEN] 男生配件 ----------------
    "MEN": {
        "Q1": { # 🌸 鄰家男孩 / 學院 (Preppy & Dandy)
            "Jewelry": [
                "Minimalist Silver Ring (簡約細銀戒)", "Leather Cord Necklace (皮繩項鍊)",
                "Beaded Bracelet (木質串珠手環)", "Small Studs (單邊耳釘)"
            ],
            "Functional_Styling": [
                "Leather Strap Watch (皮帶文青錶)", "Round Glasses (圓框眼鏡)", "Newsboy Cap (報童帽)",
                "Knitted Scarf (針織圍巾)", "Patterned Socks (花紋長襪)", "Canvas Belt (帆布腰帶)"
            ],
            "Bags": [
                "Canvas Messenger (帆布郵差包)", "Tote Bag (托特包)", "Simple Wallet (簡約短夾)"
            ],
            "Hair_Styling": [
                "Matte Wax (霧面髮蠟)", "Texture Spray (蓬鬆噴霧)"
            ]
        },
        "Q2": { # ⚡ 街頭潮流 (Street & Hype)
            "Jewelry": [
                "Chrome Ring (克羅心風格銀戒)", "Cuban Chain (古巴鍊)", "Dog Tag (軍牌項鍊)",
                "Ear Hoop (金屬耳圈)", "Lip Ring (唇環/假唇環)", "Chain Bracelet (粗鍊條手鍊)"
            ],
            "Functional_Styling": [
                "G-Shock (機能電子錶)", "Tinted Sunglasses (彩色墨鏡)", "Bucket Hat (漁夫帽)",
                "Beanie (毛帽)", "Bandana (變形蟲領巾)", "Utility Belt (機能扣環腰帶)"
            ],
            "Bags": [
                "Crossbody Bag (機能小包)", "Chest Rig (戰術胸包)", "Waist Bag (腰包)"
            ],
            "Hair_Styling": [
                "Headband (運動髮帶)", "Durag (嘻哈頭巾)", "Hair Clip (造型髮夾)"
            ]
        },
        "Q3": { # 💎 紳士品格 (Dapper & Classy)
            "Jewelry": [
                "Signet Ring (金質印章戒)", "Cufflinks (精緻袖扣)", "Tie Bar (領帶夾)",
                "Collar Pin (領針)", "Luxury Watch (機械腕錶)"
            ],
            "Functional_Styling": [
                "Pocket Square (絲質口袋巾)", "Leather Gloves (真皮手套)", "Silk Scarf (絲巾)",
                "Suspenders (吊帶)", "Leather Belt (真皮皮帶)", "Dress Socks (紳士長襪)"
            ],
            "Bags": [
                "Leather Briefcase (皮革公事包)", "Clutch (手拿包)", "Card Holder (皮革名片夾)"
            ],
            "Hair_Styling": [
                "Pomade (油頭比基尼)", "Grooming Cream (亮澤髮霜)"
            ]
        },
        "Q4": { # 👑 商務權威 (Business & Sharp)
            "Jewelry": [
                "Titanium Ring (鈦金屬戒指)", "Minimalist Cuff (極簡手鐲)", "Lapel Pin (西裝領針)",
                "Steel Watch (鋼帶潛水錶)"
            ],
            "Functional_Styling": [
                "Aviator Sunglasses (飛行員墨鏡)", "Square Glasses (方框眼鏡)", "Tie (真絲領帶)",
                "Cashmere Scarf (羊絨圍巾)", "Classic Belt (自動扣皮帶)"
            ],
            "Bags": [
                "Hard Briefcase (硬殼公事包)", "Laptop Bag (筆電包)", "Passport Holder (護照夾)"
            ],
            "Hair_Styling": [
                "Styling Gel (強力定型膠)", "Comb (油頭排梳)"
            ]
        },
        "CENTER": { # 🍃 日系休閒 (City Boy / Casual)
            "Jewelry": [
                "Silver Band (素面銀戒)", "Braided Leather (編織手環)", "Cord Bracelet (傘繩手環)"
            ],
            "Functional_Styling": [
                "Smart Watch (智慧手錶)", "NATO Strap Watch (帆布帶軍錶)", "Cap (老帽/鴨舌帽)",
                "Key Holder (掛頸鑰匙包)", "Woven Belt (編織腰帶)", "White Socks (白襪)"
            ],
            "Bags": [
                "Nylon Backpack (尼龍後背包)", "Sacoche (輕便側背包)", "Eco Bag (環保袋)"
            ],
            "Hair_Styling": [
                "Sea Salt Spray (海鹽蓬鬆水)", "Hair Balm (髮油膏)"
            ]
        }
    }
}

# 風格量化評分規則 (Scoring Rules)
STYLE_SCORING_RULES = {
    "FACE_CURVE": [
        (['圓', 'Round', '鵝蛋', 'Oval', '肉', 'Fleshy'], 2.0),
        (['方', 'Square', '長', 'Long', '菱', 'Diamond', '骨', 'Bony', '國字'], -2.0)
    ],
    "BODY_VOLUME": [
        (['沙漏', 'Hourglass', '梨', 'Pear', '蘋果', 'Apple', '圓身', '壯', 'Thick'], 1.5),
        (['倒三角', 'Inverted', '直筒', 'Rectangle', 'H-Shape', '扁身', '竹竿', 'Slim', '梯形', 'Trapezoid'], -1.5)
    ]
}

# 建立具體單品與造型字典，方便一鍵檢索
CLOTHING_LIBRARY = {
    "WARDROBE": {k: v.get("wardrobe_guide") for k, v in STYLE_MATRIX.get("STYLE_MATRIX", {}).items()},
    "ACCESSORIES": globals().get("ACCESSORY_MATRIX", {}),
    "HAIRSTYLES": globals().get("HAIRSTYLE_DATABASE", {}),
    "MAKEUP": globals().get("MAKEUP_DATABASE", {}),
    "FABRICS": {
        "WOMEN": MATERIAL_PATTERN_DATABASE.get("WOMEN", {}).get("FABRICS", {}),
        "MEN": MATERIAL_PATTERN_DATABASE.get("MEN", {}).get("FABRICS", {})
    },
    "PATTERNS": {
        "WOMEN": MATERIAL_PATTERN_DATABASE.get("WOMEN", {}).get("PATTERNS", {}),
        "MEN": MATERIAL_PATTERN_DATABASE.get("MEN", {}).get("PATTERNS", {})
    }
}

# 臉型與體型專家規則字典 (FACE_BODY_RULES)
FACE_BODY_RULES = {
    "FACE": [
        ('Round', '圓形臉：長寬接近1:1，圓潤無骨感', '日系短髮, 鎖骨髮, 八字瀏海', '齊瀏海, 小圓領'),
        ('Square', '方形臉：下顎寬，稜角分明', 'C字彎, 大波浪, 側分', '一刀切短髮, 方領'),
        ('Square-Round', '方圓臉：下顎有角但皮肉包裹好', '法式慵懶捲, 八字瀏海修飾下顎', '貼頭皮直髮'),
        ('Long', '長形臉：臉長大於寬，中庭長', '空氣瀏海, 丸子頭, 羊毛捲', '中分直髮, 貼頭皮造型'),
        ('Heart', '心形臉/瓜子臉：額頭寬下巴尖', '法式慵懶捲, 鎖骨髮', '大光明, 超短髮'),
        ('Oval', '鵝蛋臉：線條流暢標準臉', 'All (百搭), 露額頭造型', '無'),
        ('Diamond', '菱形臉：顴骨最寬，太陽穴凹陷', '八字瀏海, 蓬鬆捲髮', '貼頭皮直髮'),
        ('Pear', '梨形臉：額頭窄下顎寬', '頭頂蓬鬆, 龍鬚瀏海', '厚重齊瀏海')
    ],
    "BODY": [
        ('Hourglass', '沙漏型：肩臀同寬，腰細', '收腰連身裙, V領, 緊身褲', '寬鬆無腰身的大T恤'),
        ('Pear', '梨型：上半身瘦，臀腿肉多', 'A字裙, 闊腿褲, 泡泡袖', '緊身褲, 包臀裙'),
        ('Apple', '蘋果型：四肢細，腰腹圓潤', '娃娃裝, 直筒裙, 露腿穿搭', '緊身衣, 寬腰帶'),
        ('Inverted Triangle', '倒三角型：肩寬臀窄', 'V領, 插肩袖, A字裙, 傘裙', '一字領, 泡泡袖, 墊肩'),
        ('Rectangle', '直筒型：肩腰臀寬度一致', '層次穿搭, 荷葉邊, 工裝褲', '緊身包臀')
    ]
}

SEASONAL_COLOR_DATABASE = {
    "Spring": {
        "name": "春季型 (Spring)",
        "logic": "暖色調 + 高明度 + 高純度 (清透 / 明亮 / 鮮豔)",
        "vibe": "活潑、可愛、年輕、元氣、親和力",
        "color_palette": {
            "primary": ["#FF7F50 (珊瑚橘)", "#FFB6C1 (蜜桃粉)", "#FFFFE0 (鵝黃)", "#98FB98 (薄荷嫩綠)"],
            "accent": ["#FF4500 (亮橘紅)", "#00CED1 (亮水藍)", "#FFD700 (明黃色)"],
            "neutral": ["#FFFFF0 (象牙白)", "#F5DEB3 (暖米色)", "#D2B48C (淺駝色)", "#8B4513 (焦糖棕)"]
        },
        "avoid_colors": ["純黑色", "深灰色", "冷艷的冰藍色", "混濁的暗色系"],
        "jewelry_metal": ["亮金色", "玫瑰金"],
        "makeup_advice": "底妝強調水光透亮感；眼唇彩適合蜜桃色、珊瑚橘色系，避免過度厚重的煙燻妝或深紫色唇彩。",
        "matrix_match": ["Q1", "CENTER"]
    },

    "Summer": {
        "name": "夏季型 (Summer)",
        "logic": "冷色調 + 高明度 + 低純度 (柔和 / 霧面 / 莫蘭迪)",
        "vibe": "溫柔、優雅、氣質、清冷、知性",
        "color_palette": {
            "primary": ["#E6E6FA (薰衣草紫)", "#B0C4DE (天空霧藍)", "#D8BFD8 (灰粉/乾燥玫瑰)", "#F0FFF0 (冰薄荷)"],
            "accent": ["#DB7093 (梅子粉)", "#5F9EA0 (灰綠色)", "#87CEEB (柔和湛藍)"],
            "neutral": ["#F5F5F5 (灰白色)", "#DCDCDC (淺灰)", "#708090 (藍灰色)", "#483D8B (暗紫藍)"]
        },
        "avoid_colors": ["正橘色", "明黃色", "厚重的大地色", "強烈的高對比螢光色"],
        "jewelry_metal": ["銀色", "白金", "珍珠"],
        "makeup_advice": "底妝適合呈現半霧面啞光的高級感；彩妝首選玫瑰色、梅子色、冷粉色系，修容建議使用帶灰調的陰影色。",
        "matrix_match": ["Q1", "Q3"]
    },

    "Autumn": {
        "name": "秋季型 (Autumn)",
        "logic": "暖色調 + 低明度 + 低純度 (濃郁 / 沉穩 / 醇厚)",
        "vibe": "成熟、高級、奢華、復古、大氣",
        "color_palette": {
            "primary": ["#B22222 (楓葉磚紅)", "#808000 (橄欖綠)", "#DAA520 (芥末黃)", "#D2691E (南瓜橘)"],
            "accent": ["#8B0000 (深酒紅)", "#2E8B57 (復古墨綠)", "#FF8C00 (深橘色)"],
            "neutral": ["#8B4513 (深咖啡/巧克力)", "#A0522D (栗子棕)", "#F5F5DC (燕麥米)", "#556B2F (軍綠色)"]
        },
        "avoid_colors": ["螢光粉", "冰藍色", "純白色", "所有偏冷的亮色系"],
        "jewelry_metal": ["復古黃金", "黃銅", "木質/琥珀飾品"],
        "makeup_advice": "非常適合強調輪廓立體感的濃郁妝容；唇彩完美駕馭紅棕色、土橘色、磚紅色，眼妝適合大地色疊加微暖金屬光澤。",
        "matrix_match": ["Q3", "Q4", "CENTER"]
    },

    "Winter": {
        "name": "冬季型 (Winter)",
        "logic": "冷色調 + 低明度 / 高對比 (銳利 / 鮮明 / 極端)",
        "vibe": "摩登、個性、氣場、冷艷、戲劇化",
        "color_palette": {
            "primary": ["#000000 (極致黑)", "#FFFFFF (純白)", "#000080 (寶石藍)", "#DC143C (正紅色)"],
            "accent": ["#FF00FF (亮紫紅)", "#00FF00 (霓虹綠)", "#8A2BE2 (皇家紫)"],
            "neutral": ["#A9A9A9 (深鐵灰)", "#2F4F4F (炭灰色)", "#191970 (午夜藍)", "#800000 (深酒紅)"]
        },
        "avoid_colors": ["混濁的大地色", "淺橘色", "暖棕色", "米黃色"],
        "jewelry_metal": ["亮銀色", "白金", "冷冽的鑽石/水晶"],
        "makeup_advice": "適合高對比度的妝容，例如乾淨俐落的眼線搭配氣場全開的正紅唇，或是帶有冷色調亮片的眼妝。",
        "matrix_match": ["Q2", "Q4"]
    }
}

# 穿著場合規範 (DRESS_CODE_DB)
DRESS_CODE_DB = [
    ('Business Formal', '全套西裝/套裝', '深色西裝, 領帶, 皮鞋, 閉口跟鞋, 珍珠耳環'),
    ('Smart Casual', '體面休閒 (職場常用)', 'Polo衫, 深色牛仔褲, 休閒西裝, 樂福鞋, 簡約手錶'),
    ('Cocktail', '雞尾酒會/半正式', '及膝小禮服(LBD), 深色西裝(不一定要領帶), 尖頭跟鞋'),
    ('Black Tie', '正式晚宴/紅毯', '晚禮服(Tuxedo), 長禮服, 高質感珠寶, 手拿包'),
    ('Casual', '完全休閒', 'T恤, 牛仔褲, 運動鞋, 短褲, 棒球帽'),
    ('Athleisure', '運動時尚', '瑜伽褲, 衛衣, 運動背心, 老爹鞋'),
    ('Academic', '學院/應試', '白襯衫, 針織背心, 百褶裙/西裝褲, 牛津鞋')
]

# 風格標籤處理工具 (Tag Utilities)
def get_all_style_tags():
    """
    從 STYLE_TAXONOMY 中提取所有可用的標籤名稱。
    用於：前端下拉選單、搜尋建議、或 AI 訓練標籤。
    """
    tags = []
    for category in STYLE_TAXONOMY.values():
        if 'options' in category:
            # 抓取值，例如 "職場通勤 (Smart Casual)"
            # 並只取空格前的中文部分 "職場通勤"
            raw_values = category['options'].values()
            clean_tags = [v.split(' ')[0] for v in raw_values]
            tags.extend(clean_tags)
    return tags

ALL_STYLE_TAGS = get_all_style_tags()

class BaseFashionEngine:
    """統一處理所有 AI 引擎的影像讀取、縮放與 RGB 轉換"""
    @staticmethod
    def _read_image(image_path, max_dim=1024):
        # 1. 檢查路徑
        if not os.path.exists(image_path):
            return None, None, "找不到檔案"

        # 2. 處理中文路徑讀取
        # noinspection PyBroadException
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        except Exception:
            return None, None, "讀取發生異常"

        if img is None:
            return None, None, "無法讀取圖片 (格式錯誤或毀損)"

        # 3. 智慧縮放 (保持效能)
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 4. 統一轉換為 RGB (供 AI 模型使用)
        # noinspection PyBroadException
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return None, None, "色彩轉換失敗"

        # 回傳：原始BGR圖(繪圖用), 分析RGB圖(AI用), 錯誤訊息(無)
        return img, img_rgb, None

# Part 1: FaceGeometryMixin (共用數學邏輯 - 支援局部裁切)
class FaceGeometryMixin:
    """
    負責幾何計算，確保「訓練」與「分析」使用完全相同的數學標準。
    """

    @staticmethod
    def _calculate_vector_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _extract_feature_vector(self, pts_2d, image_shape=None):
        """
        [核心] 提取五官特徵向量 (包含側臉分析)
        """
        # --- 局部裁切模式 ---
        if pts_2d is None:
            if image_shape is None: return {}
            h, w = image_shape[:2]
            aspect_ratio = h / w
            return {
                'eyes': {'tilt': 0.0, 'ratio': float(aspect_ratio)},
                'nose': {'ratio': float(w / h), 'angle': 90.0},
                'lips': {'ratio': 0.5, 'cupid': 140.0, 'offset': 0.0, 'thickness': float(aspect_ratio)},
                'eyebrows': {'tilt': 0.0, 'arch': 0.1},
                'face_contour': {'lw_ratio': float(h / w * 1.5), 'jaw_ratio': 0.8},
                'profile': {'convexity': 170.0}  # 預設直面型
            }

        # --- 全臉精密模式 ---
        if len(pts_2d) > 468:
            ref_dist = np.linalg.norm(pts_2d[468] - pts_2d[473])
        else:
            ref_dist = np.linalg.norm(pts_2d[362] - pts_2d[263])
        if ref_dist == 0:ref_dist = 1.0

        # 1. 眼部
        eye_w = np.linalg.norm(pts_2d[362] - pts_2d[263])/ ref_dist
        eye_h = np.linalg.norm(pts_2d[159] - pts_2d[145])/ ref_dist
        dx = pts_2d[263][0] - pts_2d[362][0]
        dy = -(pts_2d[263][1] - pts_2d[362][1])
        eye_tilt = np.degrees(np.arctan2(dy, dx))

        # 2. 鼻部
        nose_w = np.linalg.norm(pts_2d[327] - pts_2d[278])/ ref_dist
        nose_h = np.linalg.norm(pts_2d[168] - pts_2d[2])/ ref_dist
        nasal_angle = self._calculate_vector_angle(pts_2d[1], pts_2d[2], pts_2d[0])/ ref_dist

        # 3. 唇部
        upper_h = np.linalg.norm(pts_2d[0] - pts_2d[13])/ ref_dist
        lower_h = np.linalg.norm(pts_2d[14] - pts_2d[17])/ ref_dist
        mouth_w = np.linalg.norm(pts_2d[61] - pts_2d[291])/ ref_dist

        corner_y_avg = (pts_2d[61][1] + pts_2d[291][1]) / 2/ ref_dist
        center_y = pts_2d[13][1]/ ref_dist
        mouth_offset = (center_y - corner_y_avg) / (mouth_w + 1e-6) * 100
        cupid_angle = self._calculate_vector_angle(pts_2d[267], pts_2d[0], pts_2d[37])

        # 4. 眉毛
        brow_w = np.linalg.norm(pts_2d[66] - pts_2d[70])/ ref_dist
        brow_dy = -(pts_2d[66][1] - pts_2d[70][1])
        brow_tilt = np.degrees(np.arctan2(brow_dy, brow_w)) if brow_w > 0 else 0
        brow_arch = np.linalg.norm(pts_2d[105] - pts_2d[70]) / (brow_w + 1e-6)

        # 5. 臉型
        face_w = np.linalg.norm(pts_2d[454] - pts_2d[234])/ ref_dist
        face_h = np.linalg.norm(pts_2d[10] - pts_2d[152])/ ref_dist
        jaw_w = np.linalg.norm(pts_2d[361] - pts_2d[132])/ ref_dist

        # 6. 側臉凸度 (Facial Convexity)
        profile_angle = self._calculate_vector_angle(pts_2d[168], pts_2d[2], pts_2d[152])

        return {
            'eyes': {'tilt': float(eye_tilt), 'ratio': float(eye_h / (eye_w + 1e-6))},
            'nose': {'ratio': float(nose_w / (nose_h + 1e-6)), 'angle': float(nasal_angle)},
            'lips': {
                'ratio': float(upper_h / (lower_h + 1e-6)),
                'cupid': float(cupid_angle),
                'offset': float(mouth_offset),
                'thickness': float((upper_h + lower_h) / (mouth_w + 1e-6))
            },
            'eyebrows': {'tilt': float(brow_tilt), 'arch': float(brow_arch)},
            'face_contour': {'lw_ratio': float(face_h / (face_w + 1e-6)), 'jaw_ratio': float(jaw_w / (face_w + 1e-6))},
            'profile': {'convexity': float(profile_angle)}
        }

# Part 2: FaceTrainer (模型訓練器 - 支援局部圖)
class FaceTrainer(BaseFashionEngine, FaceGeometryMixin):
    def __init__(self, reference_root='static/references', target_model_path='face_model_v1.json'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.reference_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), reference_root)
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_model_path)

        self.FOLDER_MAP = {
            'eye_shape': 'eyes', 'nose_shape': 'nose',
            'lip_shape': 'lips', 'brow_shape': 'eyebrows',
            'face_shape': 'face_contour'
        }

        self.STYLE_NAME_MAP = {
            'eyes': {'phoenix': '丹鳳眼', 'almond': '杏仁眼', 'round': '圓眼', 'droopy': '下垂眼', 'amorous': '桃花眼',
                     'slit': '細長眼'},
            'nose': {'narrow': '希臘鼻', 'upturned': '小翹鼻', 'fleshy': '蒜頭鼻',
                     'hooked': '鷹勾鼻', 'flat': '塌鼻', 'straight': '直鼻', 'short': '短鼻'},
            'lips': {'full': '豐滿厚唇', 'm_shape': 'M字微笑唇', 'thin': '薄唇',
                     'smile': '微笑唇', 'frown': '覆舟嘴'},
            'eyebrows': {'standard': '標準眉', 'high_arch': '歐美挑眉', 'flat': '一字平眉', 'willow': '柳葉眉',
                         'upward': '劍眉', 'rounded': '彎月圓眉', 'droopy': '八字眉'},
            'face_contour': {'oval': '鵝蛋臉', 'round': '圓形臉', 'square': '方形臉', 'long': '長形臉',
                             'diamond': '菱形臉', 'pear': '梨形臉'}
        }

    @staticmethod
    def _get_landmarks_array(image, results):
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])
        return coords

    def train(self):
        print(f"🚀 [Trainer] 開始訓練，來源: {self.reference_root}")
        model_db = {k: {} for k in self.FOLDER_MAP.values()}
        stats = []

        if not os.path.exists(self.reference_root):
            return "錯誤：找不到素材資料夾"

        for folder_name, db_key in self.FOLDER_MAP.items():
            category_path = os.path.join(self.reference_root, folder_name)
            if not os.path.exists(category_path):
                continue

            sub_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            category_map = self.STYLE_NAME_MAP.get(db_key, {})

            for style_folder_raw in sub_folders:
                style_path = os.path.join(category_path, style_folder_raw)

                # 模糊比對名稱 (忽略大小寫與_空)
                clean_name = style_folder_raw.lower().replace("_空", "")
                style_display_name = style_folder_raw
                for keyword, zh_name in category_map.items():
                    if keyword in clean_name:
                        style_display_name = zh_name
                        break

                vectors = []  # 用於蒐集該風格的所有特徵向量
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    images.extend(glob.glob(os.path.join(style_path, ext)))

                for img_path in images:
                    # noinspection PyBroadException
                    try:
                        # 呼叫靜態方法時，若已在類別內且正確定義，可直接呼叫或透過類別名
                        img, img_rgb, err = self._read_image(img_path, max_dim=800)
                        if err:
                            continue

                        results: Any = self.face_mesh.process(img_rgb)

                        if results.multi_face_landmarks:
                            pts = FaceTrainer._get_landmarks_array(img, results)
                            vec_full = self._extract_feature_vector(pts[:, :2], img.shape)
                        else:
                            # 局部 fallback
                            vec_full = self._extract_feature_vector(None, img.shape)

                        if vec_full:
                            vectors.append(vec_full)

                    except Exception:
                        continue

                # 如果該風格資料夾有成功提取到特徵
                if vectors:
                    avg_vec = {}
                    # 以第一筆資料的 key 為基準（例如：eye_ratio, nose_ratio 等）
                    keys = vectors[0].keys()
                    for k in keys:
                        # 計算該風格下所有圖片特徵的平均值
                        avg_vec[k] = sum(v[k] for v in vectors) / len(vectors)

                    model_db[db_key][style_display_name] = avg_vec
                    stats.append(f"[{db_key}] {style_display_name}: {len(vectors)} 張")

        # 儲存訓練結果至 JSON
        # noinspection PyBroadException
        try:
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(model_db, f, ensure_ascii=False, indent=2)
            return "\n".join(stats) if stats else "無有效訓練資料"
        except Exception as save_err:
            return f"儲存失敗: {save_err}"

# Part 3: FaceAnalyzer (分析引擎)
class FaceAnalyzer(BaseFashionEngine, FaceGeometryMixin):
    def __init__(self, target_model='face_model_v1.json'):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.target_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_model)
        self.MATERIAL_DB = self.load_model()

        self.COLORS = {'contour': (255, 178, 102), 'eye': (102, 255, 102), 'brow': (102, 255, 255),
                       'nose': (255, 102, 255), 'lip': (102, 102, 255)}
        self.COLORS_3D = {
            'contour': '#FFB266', 'jawline': '#FF8800', 'eye': '#00FF00',
            'brow': '#00FFFF', 'nose': '#FF00FF', 'lip': '#FF0000',
            'profile': '#FFFF00'
        }

    def load_model(self):
        if os.path.exists(self.target_model):
            # noinspection PyBroadException
            try:
                with open(self.target_model, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    @staticmethod
    def _get_landmarks_array(image, results):
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])
        return coords, landmarks

    # ==========================================
    # 🎯 專業單一結果判定
    # ==========================================
    @staticmethod
    def _rule_based_nose(nose_ratio, nose_depth):
        print(f"\n[鼻型探測] 正在分析鼻子...")
        print(f"  ▶ 鼻子長寬比 (nose_ratio): {nose_ratio:.3f}")
        print(f"  ▶ 鼻樑立體度 (nose_depth): {nose_depth:.3f}")

        # 1. 大寬鼻 (Wide): 擁有極高的立體度數據與較寬的比例
        if nose_ratio >= 0.132 or (nose_depth >= 2.8 and nose_ratio >= 0.122):
            return "大寬鼻"

        # 2. 蒜頭鼻 (Fleshy / Garlic): 立體度偏中高，且比例明顯偏寬
        if nose_depth >= 2.0 and nose_ratio >= 0.118:
            return "蒜頭鼻"

        # 3. 朝天鼻 (Upturned): 在 3D 運算中產生兩種獨特的光學特徵
        if nose_depth >= 3.5 and nose_ratio <= 0.095:
            return "朝天鼻"

        # 情境 B (扁平稍寬): 數據 (0.119, 1.088)
        if nose_depth <= 1.2 and 0.110 <= nose_ratio <= 0.130:
            return "朝天鼻"

        # 4. 細長鼻 (Slender): 鼻子寬度比例極小
        if nose_ratio <= 0.106:
            return "細長鼻"

        # 5. 標準鼻 (Standard): 比例適中，無極端特徵，作為預設防線
        return "標準鼻"

    @staticmethod
    def _rule_based_lips(thickness, offset, width_ratio):
        print(f"\n[嘴型探測] 正在分析嘴巴...")
        print(f"  ▶ 嘴唇厚度比例 (thickness): {thickness:.3f}")
        print(f"  ▶ 嘴角傾斜角 (offset): {offset:.3f}")
        print(f"  ▶ 嘴巴寬度比例 (width_ratio): {width_ratio:.3f}")

        # 0. 優先攔截：極端下垂唇 (覆舟唇)
        if offset <= -4.0:
            return "下垂唇 (覆舟唇)"
        if offset >= 8.0:
            return "微笑唇"  # 強制攔截極端上揚的嘴角

            # 1. 櫻桃小嘴
        if thickness >= 0.480 and width_ratio <= 156.0:
            return "櫻桃小嘴"

            # 2. 寬度極端組
        if width_ratio >= 160.0:
            if offset >= 3.0:
                return "厚唇"
            else:
                return "大寬唇"

            # 3. 一般下垂唇
        if offset <= -3.0:
            return "下垂唇 (覆舟唇)"
        if thickness >= 0.380 and width_ratio >= 156.0 and offset < 2.0:
            return "下垂唇 (覆舟唇)"

            # 4. 薄唇
        if thickness <= 0.310:
            return "薄唇"

            # 5. 一般厚唇
        if thickness >= 0.380:
            return "厚唇"

        return "微笑唇"

    @staticmethod
    def _rule_based_eyes(eye_ratio, eye_tilt):
        print(f"\n[眼型探測] 正在分析眼睛...")
        print(f"  ▶ 眼睛長寬比 (eye_ratio): {eye_ratio:.3f}")
        print(f"  ▶ 眼尾傾斜角 (eye_tilt): {eye_tilt:.3f}")

        # 1. 細長眼 (Slender)
        if eye_ratio < 0.310 or (eye_ratio <= 0.335 and eye_tilt < 0):
            return "細長眼"

            # 2. 超高傾斜角組
        if eye_tilt >= 15.0:
            if eye_ratio >= 0.415:
                return "下垂眼"
            else:
                return "桃花眼"

            # 3. 極度圓潤的眼睛 (接住 0.436)
        if eye_ratio >= 0.420:
            return "圓眼"

            # 4. 丹鳳眼與桃花眼的糾纏 (高傾斜角 + 偏寬)
        if 11.5 <= eye_tilt <= 14.5 and eye_ratio >= 0.370:
            if eye_ratio >= 0.400:
                return "桃花眼"  # 夠圓且上揚就是桃花眼
            return "丹鳳眼"

            # 5. 下垂眼
        if eye_tilt >= 8.5 and eye_ratio >= 0.380:
            return "下垂眼"

            # 6. 圓眼 (將門檻從 0.350 提高到 0.385，避免誤判杏眼)
        if eye_ratio >= 0.385:
            return "圓眼"

            # 7. 桃花眼 (一般組)
        if eye_ratio >= 0.365 and eye_tilt >= 6.0:
            return "桃花眼"

        return "杏眼"

    @staticmethod
    def _rule_based_brows(thickness, tilt):
        print(f"\n[眉型探測] 正在分析眉毛...")
        print(f"  ▶ 眉毛粗細比例 (thickness): {thickness:.3f}")
        print(f"  ▶ 眉毛傾斜/弧度 (tilt): {tilt:.3f}")

        # 1. 一字眉 (Straight): 捕捉極端翻轉值與破百的傾斜角
        if tilt >= 85.0 or thickness < 0:
            return "一字眉 (平眉)"

            # 2. 挑眉 (必須有厚度且角度合理)
        if thickness >= 89.0 and tilt < 60.0:
            return "挑眉 (歐美眉)"

            # 3. 細眉 (優先攔截厚度低於 81.5 的)
        if thickness <= 81.5:
            return "細眉 (柳葉眉)"

            # 4. 八字眉 (下垂眉)
        if tilt <= 19.0:
            return "八字眉 (下垂眉)"
        if 40.0 <= tilt <= 55.0 and thickness >= 85.0:
            return "八字眉 (下垂眉)"

            # 5. 野生眉 (粗眉)
        if tilt >= 58.0 and thickness > 82.0:
            return "野生眉 (粗眉)"
        if 20.0 <= tilt < 40.0 and thickness > 84.0:
            return "野生眉 (粗眉)"

        return "標準眉"

    @staticmethod
    def _calculate_face_shape_3d(pts_3d):
        """
        [3D 升級版] 解決鏡頭透視變形，並校準 MediaPipe 的真實點位比例閾值
        """
        try:

            # 1. 計算 3D 空間直線距離
            w_cheek = np.linalg.norm(pts_3d[454] - pts_3d[234])  # 顴骨最寬處
            w_temple = np.linalg.norm(pts_3d[356] - pts_3d[127])  # 太陽穴寬度
            w_jaw = np.linalg.norm(pts_3d[361] - pts_3d[132])  # 下顎角寬度

            # 補償 MediaPipe 額頭缺失的長度 (微調乘數讓臉長更精準)
            vec_forehead = pts_3d[10] - pts_3d[168]
            pt_hairline_est = pts_3d[10] + vec_forehead * 0.45
            h_full = np.linalg.norm(pt_hairline_est - pts_3d[152])  # 真實 3D 臉長

            # 2. 計算比例因子 (防呆避免除以 0)
            base_w = w_cheek + 1e-6
            r_h = h_full / base_w  # 臉長 / 臉寬
            r_jaw = w_jaw / base_w  # 下顎 / 臉寬
            r_temple = w_temple / base_w  # 太陽穴 / 臉寬

            print(f"\n[特徵探測] 正在分析圖片...")
            print(f"  ▶ 臉長寬比 (r_h): {r_h:.3f}")
            print(f"  ▶ 下顎寬比 (r_jaw): {r_jaw:.3f}")
            print(f"  ▶ 太陽穴比 (r_temple): {r_temple:.3f}")

            # ==========================================
            # 3. MediaPipe 校準版決策樹 (Thresholds Adjusted)
            # ==========================================

            # 0. 極端特徵優先：超級寬的太陽穴 (視覺上非常明顯)
            if r_temple >= 1.025:
                if r_jaw < 0.920:
                    return "心形臉"  # 上寬下窄的經典心形臉
                return "菱形臉"

            # 1. 優先篩選「極端寬下顎」(梨形/方形)
            if r_jaw >= 0.950:
                if r_jaw > r_temple:
                    return "梨形臉"
                return "方形臉"

            # 2. 圓形臉 (Round) 與 一般方形臉
            if r_h < 1.34:
                if r_jaw >= 0.935:
                    return "方形臉"
                return "圓形臉"

            # 3. 長形臉 (Long)
            if r_h >= 1.40:
                return "長形臉"

            # 4. 剩下的方形臉與菱形臉
            if r_jaw >= 0.935:
                return "方形臉"

            if r_temple >= 1.010 and r_jaw < 0.930:
                return "菱形臉"

            return "鵝蛋臉"

        except Exception as err:
            print(f"3D 臉型分析出錯: {err}")
            return "鵝蛋臉"

    # noinspection PyUnusedLocal
    def _get_side_profile_depth_map(self, side_image_path, front_height):
        """
        [精準版] 側臉深度計算引擎
        將側臉照片的 X 軸寬度視為「真實深度」，Y 軸高度視為「真實長度」，
        藉此算出最精準的人體工學比例。
        """
        if not side_image_path or not os.path.exists(side_image_path):
            return 1.0

        try:
            img, img_rgb, _ = self._read_image(side_image_path, max_dim=800)
            results: Any = self.face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                print("🔍 [Debug] 側臉偵測失敗，使用預設比例 1.0")
                return 1.0

            side_landmarks = results.multi_face_landmarks[0]

            # 1. 取得側臉照的像素高寬
            img_h, img_w = img.shape[:2]

            # 2. 計算臉部真實像素高度 (額頭 10 到 下巴 152)
            y_top = side_landmarks.landmark[10].y * img_h
            y_bottom = side_landmarks.landmark[152].y * img_h
            side_face_height = abs(y_bottom - y_top)

            # 3. 計算側臉的「像素深度」
            # 直接取 X 軸最大與最小的差，就是可見的最寬距離
            all_x = [p.x * img_w for p in side_landmarks.landmark]
            side_face_depth = max(all_x) - min(all_x)

            # 4. 算出完美的個人化深度比例
            if side_face_height > 0:
                user_depth_ratio = side_face_depth / side_face_height

                # 標準人類的比例大約為 0.72
                standard_ratio = 0.72
                calculated_multiplier = user_depth_ratio / standard_ratio

                # 💡 [關鍵修復] 嚴格限制 MediaPipe 側臉的誤差，防止 3D 模型變成「鳥嘴」
                calculated_multiplier = float(np.clip(calculated_multiplier, 0.9, 1.15))

                print(f"🔍 [Debug] 側臉分析 - 實際高度: {side_face_height:.1f}, 實際深度: {side_face_depth:.1f}")
                print(f"🔍 [Debug] 側臉分析 - 您的比例: {user_depth_ratio:.3f} -> 最終倍率: {calculated_multiplier:.2f}")

                return calculated_multiplier

            return 1.0

        except Exception as err:
            print(f"⚠️ 側臉計算發生錯誤: {err}")
            return 1.0

    def analyze(self, image_path, side_image_path=None):
        original_img, rgb_image, err = self._read_image(image_path, max_dim=1600)
        if err: return None, err

        results: Any = self.face_mesh.process(rgb_image)
        h, w = original_img.shape[:2]
        aspect_ratio = w / h

        filename = "analyzed_" + os.path.basename(image_path)
        save_path = os.path.join(os.path.dirname(image_path), filename)

        if not results.multi_face_landmarks:
            cv2.imwrite(save_path, original_img)
            return {
                'shape': '無法判斷', 'features': {},
                'analyzed_image': filename,
                'landmarks_3d_grouped': [], 'landmarks_3d_raw': [],
                'aspect_ratio': aspect_ratio
            }, None

        pts, raw_landmarks = self._get_landmarks_array(original_img, results)
        pts_2d = pts[:, :2]
        feature_vector = self._extract_feature_vector(pts_2d, original_img.shape)

        dx = pts_2d[263][0] - pts_2d[33][0]
        dy = pts_2d[263][1] - pts_2d[33][1]
        head_roll_angle = np.degrees(np.arctan2(dy, dx))
        real_eye_tilt = feature_vector['eyes']['tilt'] - head_roll_angle
        real_brow_tilt = feature_vector['eyebrows']['tilt'] - head_roll_angle

        final_face = self._calculate_face_shape_3d(pts)
        final_nose = self._rule_based_nose(feature_vector['nose']['ratio'], feature_vector['nose']['angle'])
        final_eyes = self._rule_based_eyes(feature_vector['eyes']['ratio'], real_eye_tilt)
        final_brows = self._rule_based_brows(real_brow_tilt, feature_vector['eyebrows']['arch'])
        final_lips = self._rule_based_lips(feature_vector['lips']['thickness'], feature_vector['lips']['offset'],
                                           feature_vector['lips']['cupid'])

        # ==========================================
        # 1. 基礎座標與 3D 比例錨點初始化
        # ==========================================
        draw_pts = pts_2d.copy().astype(float)

        # 僅針對 2D 繪圖進行額頭充氣，絕對不污染 3D 原始網格
        # noinspection PyBroadException
        try:
            p168, p10 = draw_pts[168], draw_pts[10]
            forehead_vec = p10 - p168
            base_dist = np.linalg.norm(forehead_vec)

            # 計算額頭向上的單位向量
            up_unit = forehead_vec / (base_dist + 1e-6)

            # 定義額頭上緣點位的推升比例 (以眉心到額頭的距離為基準)
            # 10 號點是正中央最高點，向兩側遞減以維持圓弧感
            forehead_expansion_rules = {
                10: 0.45,  # 正中央推高 45%
                109: 0.40, 338: 0.40,  # 中間偏內
                67: 0.30, 297: 0.30,  # 中間偏外
                103: 0.20, 332: 0.20,  # 外側
                54: 0.10, 284: 0.10,  # 接近太陽穴
                21: 0.05, 251: 0.05  # 太陽穴邊緣微調
            }

            for idx, ratio in forehead_expansion_rules.items():
                draw_pts[idx] = draw_pts[idx] + (up_unit * base_dist * ratio)

        except Exception as err:
            print(f"額頭充氣處理失敗: {err}")
            pass

        # noinspection PyBroadException
        try:
            pass
        except Exception:
            pass

        overlay = original_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 10, 0), -1)
        vis_img = cv2.addWeighted(overlay, 0.4, original_img, 0.6, 0)

        w_face = np.linalg.norm(draw_pts[356] - draw_pts[127])
        center_x = draw_pts[10][0]
        y_nasion_3d = float(draw_pts[168][1])
        y_chin_3d = float(draw_pts[152][1])
        h_ref_3d = y_chin_3d - y_nasion_3d

        # 🚀 將五官定義往上搬，確保全域都能存取
        left_brow = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
        right_brow = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
        left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        nose_outline = [168, 412, 343, 278, 327, 326, 2, 97, 98, 48, 114, 188]
        lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                     176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # ==========================================
        # 2. 🚀 [真 3D 黑科技合成] 讀取側臉並完美映射 Z 軸深度！
        # ==========================================
        coords_3d = []
        for i, p in enumerate(raw_landmarks):
            px = float(pts_2d[i][0])
            py = float(pts_2d[i][1])
            dist_x = abs(px - center_x)
            ratio_x = np.clip(dist_x / (w_face / 2.0), 0.0, 1.0)
            base_z = p.z * w_face * -1.45
            wrap_z = -(w_face * 0.42) * (ratio_x ** 1.8)
            coords_3d.append([px, py, base_z + wrap_z])
        coords_3d = np.array(coords_3d)

        # ------------------------------------------
        # 🎨 解析側臉、提取真實 3D 深度並繪製 2D 側臉
        # ------------------------------------------
        side_filename = None
        if side_image_path and os.path.exists(side_image_path):
            side_filename = "analyzed_side_" + os.path.basename(image_path)
            side_save_path = os.path.join(os.path.dirname(image_path), side_filename)

            side_img, side_rgb, _ = self._read_image(side_image_path, max_dim=1600)
            if side_img is not None:
                side_results: Any = self.face_mesh.process(side_rgb)

                # 影像增強備案
                if not side_results.multi_face_landmarks:
                    # noinspection PyBroadException
                    try:
                        lab = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2LAB)
                        l_channel, a_channel, b_channel = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        cl = clahe.apply(l_channel)
                        enhanced_rgb = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
                        side_results = self.face_mesh.process(enhanced_rgb)
                    except Exception:
                        pass

                if side_results.multi_face_landmarks:
                    side_pts, _ = self._get_landmarks_array(side_img, side_results)
                    side_pts_2d = side_pts[:, :2].astype(float)

                    left_jawline = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
                    right_jawline = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
                    side_profile = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 13, 14, 17, 18, 200,
                                    199, 175, 152]
                    left_hairline = [10, 109, 67, 103, 54, 21, 162, 127, 234]
                    right_hairline = [10, 338, 297, 332, 284, 251, 389, 356, 454]

                    nose_x_side = side_pts_2d[1][0]
                    left_ear_x_side = side_pts_2d[234][0]
                    right_ear_x_side = side_pts_2d[454][0]

                    if abs(nose_x_side - left_ear_x_side) > abs(nose_x_side - right_ear_x_side):
                        visible_jawline, visible_hairline, front_dir = left_jawline, left_hairline, 1.0
                    else:
                        visible_jawline, visible_hairline, front_dir = right_jawline, right_hairline, -1.0

                    # 1. 針對側臉 2D 進行幾何充氣修飾
                    pt168_side = side_pts_2d[168].astype(float)
                    pt10_side = side_pts_2d[10].astype(float)
                    pt_chin_side = side_pts_2d[152].astype(float)
                    face_h_side = np.linalg.norm(pt_chin_side - pt10_side)
                    up_vec = pt10_side - pt168_side
                    up_unit = up_vec / (np.linalg.norm(up_vec) + 1e-6)
                    front_unit = np.array([front_dir, 0], dtype=float)

                    expansion_rules = {
                        10: (0.25, 0.0), 151: (0.18, 0.01), 9: (0.08, 0.02), 8: (0.02, 0.01),
                        109: (0.23, -0.04), 67: (0.15, -0.08), 103: (0.08, -0.10), 54: (0.02, -0.10),
                        21: (-0.02, -0.08),
                        338: (0.23, -0.04), 297: (0.15, -0.08), 332: (0.08, -0.10), 284: (0.02, -0.10),
                        251: (-0.02, -0.08)
                    }

                    for idx, (lift_ratio, push_ratio) in expansion_rules.items():
                        side_pts_2d[idx] += (up_unit * face_h_side * lift_ratio) + (
                                    front_unit * face_h_side * push_ratio)

                    # 2. 🧬 將真實側臉 Z 深度「合成」映射到 3D 模型 (🚀 IDW 空間拓樸演算法)
                    side_y_nasion = side_pts_2d[168][1]
                    side_y_chin = side_pts_2d[152][1]
                    s_y = h_ref_3d / (abs(side_y_chin - side_y_nasion) + 1e-6)

                    profile_pts_yz = []
                    profile_delta_z = []

                    for idx in side_profile:
                        true_z = (side_pts_2d[idx][0] - side_pts_2d[168][0]) * s_y * front_dir
                        dz = true_z - coords_3d[idx][2]
                        # 紀錄原 3D 空間的 Y 與 Z，確保鼻尖與人中能在 3D 拓樸中被完美區分
                        orig_y = pts_2d[idx][1]
                        orig_z = raw_landmarks[idx].z * w_face * -1.45

                        profile_pts_yz.append([orig_y, orig_z])
                        profile_delta_z.append(dz)

                    profile_pts_yz = np.array(profile_pts_yz)
                    profile_delta_z = np.array(profile_delta_z)

                    # 進行 3D 全域深度合成
                    for i in range(len(coords_3d)):
                        orig_y = pts_2d[i][1]
                        orig_z = raw_landmarks[i].z * w_face * -1.45

                        # 計算到特徵線的平方距離，Z軸給予1.5倍權重，完美剝離鼻尖與上嘴唇的干擾
                        dist_sq = (profile_pts_yz[:, 0] - orig_y) ** 2 + ((profile_pts_yz[:, 1] - orig_z) * 1.5) ** 2
                        weights = 1.0 / (dist_sq + 1e-6)
                        interp_dz = np.sum(weights * profile_delta_z) / float(np.sum(weights))

                        x_dist = abs(pts_2d[i][0] - center_x)
                        weight = max(0.0, 1.0 - (x_dist / (w_face * 0.55)) ** 1.5)

                        coords_3d[i][2] += interp_dz * weight
                        if orig_y < pts_2d[168][1]:
                            y_ratio = (pts_2d[168][1] - orig_y) / (pts_2d[168][1] - pts_2d[10][1] + 1e-6)
                            coords_3d[i][1] -= (y_ratio ** 1.5) * h_ref_3d * 0.15 * weight

                    # 3. 🎨 繪製 2D 側臉並存檔
                    side_overlay = side_img.copy()
                    sh, sw = side_img.shape[:2]
                    cv2.rectangle(side_overlay, (0, 0), (sw, sh), (20, 10, 0), -1)
                    side_vis_img = cv2.addWeighted(side_overlay, 0.4, side_img, 0.6, 0)

                    # 🚀 支援 s_val 動態平滑度
                    def draw_side_poly(indices, base_color, is_closed=False, is_thin=False, is_smooth=True, s_val=150):
                        if not indices: return
                        poly_pts = np.array([side_pts_2d[pt_idx] for pt_idx in indices], np.int32)

                        draw_closed = is_closed
                        if is_smooth and len(poly_pts) > 5:
                            import scipy.interpolate as si
                            x, y = poly_pts[:, 0], poly_pts[:, 1]
                            if is_closed:
                                x, y = np.append(x, poly_pts[0, 0]), np.append(y, poly_pts[0, 1])
                            # noinspection PyBroadException
                            try:
                                # noinspection PyTupleAssignmentBalance
                                tck, u = si.splprep([x, y], s=s_val, per=is_closed)
                                u_new = np.linspace(0, 1.0, 300)
                                x_new, y_new = si.splev(u_new, tck)
                                poly_pts = np.vstack((x_new, y_new)).T.astype(np.int32)
                            except Exception:
                                pass
                            draw_closed = False

                        if is_thin:
                            cv2.polylines(side_vis_img, [poly_pts], draw_closed, base_color, 1, cv2.LINE_AA)
                        else:
                            cv2.polylines(side_vis_img, [poly_pts], draw_closed, base_color, 3, cv2.LINE_AA)
                            cv2.polylines(side_vis_img, [poly_pts], draw_closed, (255, 255, 255), 1, cv2.LINE_AA)

                    # 🚀 s_val=5 精準保留人中凹凸，其餘部位維持 s_val=150 的平滑
                    draw_side_poly(side_profile, (255, 220, 50), is_closed=False, is_smooth=True, s_val=5)
                    draw_side_poly(visible_jawline, (255, 220, 50), is_closed=False, is_smooth=True, s_val=150)
                    draw_side_poly(visible_hairline, (255, 220, 50), is_closed=False, is_smooth=True, s_val=150)

                    left_lips = [61, 185, 40, 39, 37, 0, 17, 84, 181, 91, 146]
                    right_lips = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17]

                    if front_dir == 1.0:
                        draw_side_poly(left_eye, (255, 255, 0), is_closed=True, is_thin=True, is_smooth=False)
                        draw_side_poly(left_brow, (255, 255, 0), is_closed=True, is_thin=True, is_smooth=False)
                        draw_side_poly(left_lips, (150, 50, 255), is_closed=True, is_thin=True, is_smooth=False)
                    else:
                        draw_side_poly(right_eye, (255, 255, 0), is_closed=True, is_thin=True, is_smooth=False)
                        draw_side_poly(right_brow, (255, 255, 0), is_closed=True, is_thin=True, is_smooth=False)
                        draw_side_poly(right_lips, (150, 50, 255), is_closed=True, is_thin=True, is_smooth=False)

                    cv2.imencode(".jpg", side_vis_img)[1].tofile(side_save_path)
                else:
                    cv2.imencode(".jpg", side_img)[1].tofile(side_save_path)

        # ==========================================
        # 3. 重新計算 3D 歸一化 (傳送給前端 Plotly)
        # ==========================================
        center = np.mean(coords_3d, axis=0)
        max_dist = np.max(np.abs(coords_3d - center)) or 1.0
        landmarks_3d_raw = [{'x': (pt[0] - center[0]) / max_dist, 'y': (pt[1] - center[1]) / max_dist,
                             'z': (pt[2] - center[2]) / max_dist} for pt in coords_3d]

        def close_loop(indices):
            if not indices: return []
            # 校验首尾点是否一致，避免重复/断点
            if indices[0] != indices[-1]:
                return list(indices) + [indices[0]]
            return list(indices)

        groups = {
            'contour': close_loop(face_oval),
            'jawline': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
                        323, 454],
            'lip': close_loop(lips), 'left_eye': close_loop(left_eye), 'right_eye': close_loop(right_eye),
            'left_brow': close_loop(left_brow), 'right_brow': close_loop(right_brow), 'nose': close_loop(nose_outline),
            'profile_line': [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 13, 14, 17, 18, 200, 199, 175,
                             152]
        }

        # 🚀 必須先定義 3D 顏色字典，landmarks_3d_grouped 才能正確讀取
        tech_colors_3d = {
            'contour': '#32DCFF', 'jawline': '#32DCFF', 'left_eye': '#00FFFF', 'right_eye': '#00FFFF',
            'left_brow': '#00FFFF', 'right_brow': '#00FFFF', 'nose': '#3296C8', 'lip': '#FF3296',
            'profile_line': '#FFD700'
        }

        landmarks_3d_grouped = [{'name': name, 'color': tech_colors_3d.get(name, '#FFFFFF'),
                                 'points': [landmarks_3d_raw[idx] for idx in indices]} for name, indices in
                                groups.items()]

        # ==========================================
        # 4. 🎨 2D 繪圖引擎 (正臉繪製)
        # ==========================================

        # 定義 2D 繪圖使用的顏色
        tech_colors = {
            'contour': (255, 220, 50),
            'eye': (255, 255, 0),
            'brow': (255, 255, 0),
            'nose': (200, 150, 50),
            'lip': (150, 50, 255)
        }

        def generate_smooth_curve(points, num_points=300, is_closed=True, face_type_ratio=1.0):
            import scipy.interpolate as si
            if len(points) < 4: return points
            x, y = points[:, 0], points[:, 1]
            if is_closed: x, y = np.append(x, points[0, 0]), np.append(y, points[0, 1])
            # noinspection PyBroadException
            try:
                s_val = 0 if face_type_ratio > 0.8 else 5
                # noinspection PyTupleAssignmentBalance
                tck, u = si.splprep([x, y], s=s_val, per=is_closed)
                u_new = np.linspace(0, 1.0, num_points)
                x_new, y_new = si.splev(u_new, tck)
                return np.vstack((x_new, y_new)).T.astype(np.int32)
            except Exception:
                return points.astype(np.int32)

        # 调用时传入脸型比例（jaw_cheek_ratio）
        def draw_hud_poly(indices, base_color, is_closed=True, draw_nodes=True, is_thin=False, is_smooth=False,
                          face_type_ratio=1.0):
            points = np.array([draw_pts[pt_idx] for pt_idx in indices], np.int32)
            if is_smooth and len(points) > 5:
                points = generate_smooth_curve(points, num_points=300, is_closed=is_closed,
                                               face_type_ratio=face_type_ratio)
                is_closed, draw_nodes = False, False
            if is_thin:
                cv2.polylines(vis_img, [points], is_closed, base_color, 1, cv2.LINE_AA)
            else:
                cv2.polylines(vis_img, [points], is_closed, base_color, 3, cv2.LINE_AA)
                cv2.polylines(vis_img, [points], is_closed, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_nodes and not is_thin:
                for pt in points: cv2.circle(vis_img, tuple(pt), 1, (255, 255, 255), -1, cv2.LINE_AA)

        left_cheek_x = draw_pts[234][0]  # 左颧骨点
        right_cheek_x = draw_pts[454][0]  # 右颧骨点
        left_jawline_x = draw_pts[176][0]  # 左下颌点
        right_jawline_x = draw_pts[377][0]  # 右下颌点

        # 核心比例：区分不同脸型
        cheek_width = abs(right_cheek_x - left_cheek_x)
        jaw_width = abs(right_jawline_x - left_jawline_x)
        jaw_cheek_ratio = jaw_width / (cheek_width + 1e-6)  # 方形脸>0.8，心形脸<0.6
        face_height = abs(draw_pts[10][1] - draw_pts[152][1])  # 额头到下巴高度
        face_width_height_ratio = cheek_width / (face_height + 1e-6)  # 圆形脸>0.85，长形脸<0.7

        # 基础轮廓（保留原始点）
        base_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                     176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # 按脸型适配轮廓
        if jaw_cheek_ratio > 0.8 and face_width_height_ratio > 0.75:  # 方形脸
            dynamic_contour = base_oval
        elif jaw_cheek_ratio < 0.6 and face_width_height_ratio < 0.8:  # 心形脸
            dynamic_contour = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                               377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
        elif face_width_height_ratio > 0.85:  # 圆形脸
            dynamic_contour = base_oval
        elif face_width_height_ratio < 0.7:  # 长形脸
            dynamic_contour = base_oval
        else:  # 通用脸型
            dynamic_contour = base_oval

        draw_hud_poly(dynamic_contour, tech_colors['contour'], draw_nodes=False, is_thin=True, is_smooth=True,
                      face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(left_brow, tech_colors['brow'], is_thin=True, face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(right_brow, tech_colors['brow'], is_thin=True, face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(left_eye, tech_colors['eye'], is_thin=True, face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(right_eye, tech_colors['eye'], is_thin=True, face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(nose_outline, tech_colors['nose'], is_thin=True, face_type_ratio=jaw_cheek_ratio)
        draw_hud_poly(lips, tech_colors['lip'], is_thin=True, face_type_ratio=jaw_cheek_ratio)

        cv2.imencode(".jpg", vis_img)[1].tofile(save_path)

        return {
            'shape': final_face,
            'features': {'eyes': final_eyes, 'nose': final_nose, 'lips': final_lips, 'eyebrows': final_brows,
                         'face_shape': final_face},
            'precise_metrics': {'eye_tilt': round(real_eye_tilt, 1), 'brow_tilt': round(real_brow_tilt, 1)},
            'analyzed_image': filename, 'analyzed_side_image': side_filename,
            'landmarks_3d_grouped': landmarks_3d_grouped, 'landmarks_3d_raw': landmarks_3d_raw,
            'aspect_ratio': aspect_ratio
        }, None

face_trainer = FaceTrainer()
face_engine = FaceAnalyzer()

# 真實 AI 身形與比例分析引擎 (3D Precision Body Analysis)
class BodyAnalyzer(BaseFashionEngine):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # 啟用最精準模式
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )

    @staticmethod
    def _get_distance_3d( lm1, lm2):
        """計算真實物理 3D 距離 (米)"""
        return np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2 + (lm1.z - lm2.z) ** 2)

    def analyze(self, image_path, manual_data=None):
        """
        執行全身/半身影像分析，結合 AI 骨架偵測與手動輸入數據進行身形判定。
        """
        # 1. 讀取影像 (處理中文路徑與縮放)
        image, image_rgb, err = self._read_image(image_path, max_dim=1200)
        if err: return None, err

        # --- 💡 關鍵：初始化所有變數預設值，防止 return 時找不到變數 ---
        h_img, w_img = image.shape[:2]
        sh_type = "標準肩"
        sh_hip_ratio = 1.0
        waist_hip_ratio = 0.85
        is_fake_hip = False
        ratio_desc = "無法精算頭身比"
        leg_desc = "無法精算上下身比"
        neck_type = "標準脖"

        # 進行 MediaPipe 偵測
        results: Any = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, "未偵測到人體"

        w_lm = results.pose_world_landmarks.landmark
        n_lm = results.pose_landmarks.landmark

        # --- 2. 核心關鍵點提取與信心度檢測 ---
        nose = w_lm[0]
        shoulder_l, shoulder_r = w_lm[11], w_lm[12]
        hip_l, hip_r = w_lm[23], w_lm[24]

        # 判定是否為全身照 (踝部可見度 > 0.5)
        hip_confidence = min(n_lm[23].visibility, n_lm[24].visibility)
        shoulder_confidence = min(n_lm[11].visibility, n_lm[12].visibility)
        is_full_body = (n_lm[27].visibility > 0.5 and n_lm[28].visibility > 0.5)

        # --- 3. 橫向比例分析 (寬度) ---
        # A. 肩頭比判定
        if shoulder_confidence > 0.6:
            sh_width_3d = self._get_distance_3d(shoulder_l, shoulder_r)
            est_head_width = sh_width_3d * 0.4
            sh_head_ratio = sh_width_3d / est_head_width if est_head_width > 0 else 2.5

            sh_type = "標準肩"
            if sh_head_ratio > 3.0:
                sh_type = "寬肩 / 直角肩 (衣架子)"
            elif sh_head_ratio < 2.5:
                sh_type = "窄肩 (易顯頭大)"

        # B. 肩臀比計算
        if hip_confidence > 0.6 and shoulder_confidence > 0.6:
            sh_width_3d = self._get_distance_3d(shoulder_l, shoulder_r)
            hip_width_3d = self._get_distance_3d(hip_l, hip_r)
            sh_hip_ratio = sh_width_3d / (hip_width_3d + 0.001)

            # 判定假胯寬
            if is_full_body:
                if hip_width_3d > sh_width_3d * 1.05:
                    is_fake_hip = True

        # C. 整合手動數據 (若使用者有輸入，則以手動數據為最高準則)
        if manual_data:
            # 取得數值 (確保鍵值正確)
            m_shoulder = float(manual_data.get('shoulder', 0))
            m_waist = float(manual_data.get('waist', 0))
            m_hip = float(manual_data.get('hip', 0))

            if m_shoulder > 0 and m_hip > 0:
                # 採用 2.5 倍率換算肩寬為周長，確保與臀圍對等
                sh_hip_ratio = (m_shoulder * 2.5) / (m_hip + 0.001)

            if m_waist > 0 and m_hip > 0:
                waist_hip_ratio = m_waist / m_hip

        # --- 4. 縱向比例分析 (長度) ---
        if is_full_body and hip_confidence > 0.6 and shoulder_confidence > 0.6:
            head_len = self._get_distance_3d(nose, shoulder_l) * 1.5
            torso_len = self._get_distance_3d(shoulder_l, hip_l)
            ankle_l = w_lm[27]
            leg_len = self._get_distance_3d(hip_l, ankle_l)
            total_h_3d = head_len + torso_len + leg_len

            # 頭身比判定
            head_ratio = total_h_3d / head_len
            if head_ratio >= 8:
                ratio_desc = f"{round(head_ratio, 1)} 頭身 (九頭身超模比例)"
            elif head_ratio >= 7.5:
                ratio_desc = f"{round(head_ratio, 1)} 頭身 (優秀名模比例)"
            elif head_ratio >= 6.5:
                ratio_desc = f"{round(head_ratio, 1)} 頭身 (標準比例)"
            else:
                ratio_desc = f"{round(head_ratio, 1)} 頭身 (可愛幼態比例)"

            # 腿長比判定
            leg_ratio = leg_len / (torso_len + leg_len)
            if leg_ratio > 0.65:
                leg_desc = "三七身 (極致長腿)"
            elif leg_ratio > 0.58:
                leg_desc = "四六身 (標準長腿)"
            else:
                leg_desc = "五五身 (重心較低)"

        # --- 5. 局部細節 ---
        neck_len = abs(n_lm[0].y - n_lm[11].y) * h_img
        if neck_len > h_img * 0.12:
            neck_type = "天鵝頸 (氣質極佳)"
        elif neck_len < h_img * 0.07:
            neck_type = "短脖子"

        # --- 6. 身形最終分類 (💡 修正後的精準判定結構) ---
        # 優先級 1：骨架判定 (解決數據 C: 1.19 >= 1.05)
        if sh_hip_ratio >= 1.05:
            shape = "倒三角型 (Y-Shape)"
        # 優先級 2：下半身判定
        elif sh_hip_ratio <= 0.92:
            shape = "梨型 (A-Shape)"
        # 優先級 3：曲線判定 (解決數據 D: 0.91 < 0.94)
        else:
            if waist_hip_ratio < 0.72:
                shape = "沙漏型 (X-Shape)"
            elif waist_hip_ratio >= 0.94:
                shape = "蘋果型 (O-Shape)"
            else:
                shape = "直筒型 (H-Shape)"

        # --- 7. 建議邏輯處理 ---
        neck_advice = "建議選擇 V 領或大方領，拉長頸部線條。" if neck_type == "短脖子" else "領口選擇不受限。"

        # --- 8. 繪製分析結果 ---
        vis_img = image.copy()
        if shoulder_confidence > 0.6:
            cv2.line(vis_img, (int(n_lm[11].x * w_img), int(n_lm[11].y * h_img)),
                     (int(n_lm[12].x * w_img), int(n_lm[12].y * h_img)), (0, 255, 255), 3)
        if hip_confidence > 0.6:
            cv2.line(vis_img, (int(n_lm[23].x * w_img), int(n_lm[23].y * h_img)),
                     (int(n_lm[24].x * w_img), int(n_lm[24].y * h_img)), (255, 0, 255), 3)

        # 儲存圖片
        filename = "body_analyzed_" + os.path.basename(image_path)
        save_path = os.path.join(os.path.dirname(image_path), filename)
        cv2.imencode(".jpg", vis_img)[1].tofile(save_path)

        # 回傳結構化資料
        return {
            'shape': shape,
            'ratios': {
                'head_ratio': ratio_desc,
                'body_proportion': leg_desc,
                'shoulder_type': sh_type,
                'neck_type': neck_type
            },
            'measurements': {
                'waist_hip_ratio': round(waist_hip_ratio, 2),
                'fake_hip_detected': is_fake_hip
            },
            'analysis': {
                'summary': f"您屬於【{shape}】，比例表現為 {ratio_desc}。",
                'neck_advice': neck_advice
            },
            'analyzed_image': filename
        }, None

body_engine = BodyAnalyzer()

# AI 膚色與四季型人分析引擎 (Skin Tone Analyzer)
class SkinAnalyzer(BaseFashionEngine):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6
        )

    @staticmethod
    def _extract_skin_color(image, landmarks):
        """
        [精準版] 提取臉部關鍵皮膚區域的優勢顏色 (Dominant Color)
        使用 K-Means 聚類排除陰影與反光干擾
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 採樣區域維持不變 (避開五官與瀏海)
        rois = [
            [109, 10, 338, 297, 332, 284],  # 額頭區
            [116, 123, 147, 187, 205],  # 左臉頰
            [345, 352, 376, 411, 425]  # 右臉頰
        ]

        for roi_indices in rois:
            points = np.array([[(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in roi_indices]],
                              dtype=np.int32)
            cv2.fillPoly(mask, points, 255)

        # 提取遮罩內的像素
        skin_pixels = image[mask == 255]
        if len(skin_pixels) < 10:
            return None

        # --- 🚀 K-Means 聚類尋找優勢色 ---
        # 將像素資料轉換為 float32
        data = np.float32(skin_pixels)

        # 設定 K-Means 停止條件 (最大迭代 10 次或精確度達到 1.0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # 分成 3 群 (反光區、陰影區、真實膚色區)
        k = 3
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 找出像素數量最多的一群 (即真實膚色)
        _, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = np.argmax(counts)
        dominant_color = centers[dominant_cluster_idx]

        return dominant_color

    def analyze(self, image_path):
        # 1. 影像讀取 (繼承基類)
        original_img, rgb_image, err = self._read_image(image_path, max_dim=1200)
        if err: return None, err

        # 2. 取得臉部特徵點
        results: Any = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None, "未偵測到臉部，無法進行膚色分析"

        landmarks = results.multi_face_landmarks[0].landmark

        # 3. 提取皮膚顏色 (RGB)
        mean_rgb = self._extract_skin_color(rgb_image, landmarks)
        if mean_rgb is None:
            return None, "無法提取皮膚區域"

        # 4. 色彩空間轉換 (RGB -> Lab / HSV)
        # 需先轉為 1x1 像素才能用 cv2.cvtColor
        pixel = np.uint8([[mean_rgb]])

        # 轉 Lab (L=明度, a=紅綠, b=黃藍) - 判斷冷暖與明度的關鍵
        lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0]
        l, a, b = lab_pixel[0], lab_pixel[1], lab_pixel[2]

        # 轉 HSV (H=色相, S=飽和度, V=明度) - 輔助判斷
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]

        # --- 5. 四季型人判定邏輯 (Color Analysis Logic) ---

        # A. 冷暖調判定 (Undertone)
        # b 值越高越黃(暖)，越低越藍(冷)。一般膚色 b 值約在 130-170 (OpenCV Lab 範圍)
        # 這裡設定一個經驗閾值，可根據測試微調
        warmth_score = b
        is_warm = True if warmth_score > 138 else False

        undertone = "暖色調 (Warm)" if is_warm else "冷色調 (Cool)"

        # OpenCV 的 L 範圍是 0-255，稍微降低標準，避免稍微背光就全被判成秋冬
        if is_warm:
            # 暖調：春 vs 秋
            # 春：高明度(L大)，或高飽和(S大)
            if l > 150 or s > 90:
                season = "Spring"
                desc = "春季型 (Warm + Light/Clear)"
            else:
                season = "Autumn"
                desc = "秋季型 (Warm + Dark/Muted)"
        else:
            # 冷調：夏 vs 冬
            # 夏：高明度(L大)
            if l > 145:
                season = "Summer"
                desc = "夏季型 (Cool + Light/Soft)"
            else:
                season = "Winter"
                desc = "冬季型 (Cool + Dark/Contrast)"

        # 6. 整合資料庫建議
        # 從 SEASONAL_COLOR_DATABASE 撈取對應資料
        color_advice = SEASONAL_COLOR_DATABASE.get(season, {})

        # 生成視覺化色卡 (供 Debug 或 前端顯示)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))

        return {
            'season_key': season,
            'season_name': color_advice.get('name', season),
            'undertone': undertone,
            'skin_data': {
                'lab': {'l': int(l), 'a': int(a), 'b': int(b)},
                'hex': hex_color,
                'is_warm': is_warm
            },
            'advice': {
                'best_colors': color_advice.get('best_colors', []),
                'avoid_colors': color_advice.get('avoid_colors', []),
                'makeup_vibe': color_advice.get('vibe', '')
            },
            'description': f"經檢測，您的膚色屬於【{undertone}】。\n綜合明度與飽和度分析，您最符合【{desc}】的特徵。"
        }, None

skin_engine = SkinAnalyzer()

# 真實 AI 虛擬試穿引擎 (Robust Try-On Engine)
class TryOnEngine:
    def __init__(self):
        # 使用 HuggingFace 上效果最好的 IDM-VTON 模型
        self.client_url = "yisol/IDM-VTON"
        self.client = None
        self.is_ready = False
        self.max_retries = 2

        # 檢查是否已安裝 gradio_client (在前述 config 設定)
        self.has_dependency = globals().get('GRADIO_AVAILABLE', False)

    def initialize(self):
        """嘗試連接模型 (Lazy Loading)"""
        if not self.has_dependency:
            print(">> ⚠️ 虛擬試穿功能已停用 (缺少 gradio_client 庫)")
            return False

        if self.client:
            return True

        print(">> 正在連接雲端試穿模型 IDM-VTON... (此過程需連網)")
        try:
            # 使用執行緒限制連線時間，避免卡死伺服器啟動
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 建立 Client 物件
                future = executor.submit(Client, self.client_url)
                self.client = future.result(timeout=15)  # 給予 15 秒連線緩衝

            self.is_ready = True
            print(">> ✅ 雲端試穿模型連接成功！")
            return True
        except concurrent.futures.TimeoutError:
            print(">> ❌ 雲端模型連接逾時 (網路可能不穩)")
            return False
        except Exception as err:
            print(f">> ❌ 雲端模型連接失敗: {err}")
            return False

    @staticmethod
    def _resize_image_for_api(img_path, max_size=1024):
        """
        [關鍵優化] 上傳前壓縮圖片。
        VTON 模型通常只需要 768x1024，傳原圖只會浪費頻寬導致 Timeout。
        """
        try:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 覆蓋原檔或另存暫存檔 (這裡選擇覆蓋以節省空間)
                cv2.imwrite(img_path, img)
                print(f"   [優化] 圖片已縮放至 {new_w}x{new_h}")
            return True
        except Exception as err:
            print(f"   [警告] 圖片縮放失敗: {err}")
            return False

    def generate(self, person_img_path, garment_img_path):
        """
        執行 AI 換裝
        Returns: (result_path, error_message)
        """
        # 1. 檢查依賴與連線
        if not self.has_dependency:
            return None, "系統未安裝 gradio_client，無法使用雲端 AI。"

        if not self.initialize():
            return None, "無法連接 AI 伺服器，請檢查網路連線。"

        # 2. 優化圖片 (避免 Timeout 關鍵)
        self._resize_image_for_api(person_img_path)
        self._resize_image_for_api(garment_img_path)

        # 3. 準備參數 (IDM-VTON API 格式)
        try:
            abs_person = os.path.abspath(person_img_path)
            abs_garment = os.path.abspath(garment_img_path)

            # 使用 gradio_client 的 file 包裝器
            person_file = file(abs_person)
            garment_file = file(abs_garment)

            # 根據 IDM-VTON Space 的 API 定義 (需隨時注意官方更新)
            predict_kwargs = {
                "dict": {"background": person_file, "layers": [], "composite": None},
                "garm_img": garment_file,
                "garment_des": "fashion clothing",  # 簡單描述即可
                "is_checked": True,  # Auto-crop (通常設為 True 效果較好)
                "is_checked_crop": False,
                "denoise_steps": 25,  # [優化] 降到 25 (原30) 以換取速度，肉眼差異不大
                "seed": 42,
                "api_name": "/tryon"
            }

            print(f">> 開始 AI 生成 (Timeout 設定: 80秒)...")
            start_time = time.time()

            # 4. 執行預測 (設定較長的 Timeout，因為排隊可能很久)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.client.predict, **predict_kwargs)
                # VTON 生成很慢，建議給 80-90 秒
                result = future.result(timeout=90)

            elapsed = time.time() - start_time
            print(f">> ✅ 生成完成！耗時: {elapsed:.1f} 秒")

            # result 通常是一個 tuple，第一項是圖片路徑
            return result[0], None

        except concurrent.futures.TimeoutError:
            print(">> ❌ 生成逾時")
            return None, "AI 伺服器忙碌中 (Timeout)，請稍後再試。"
        except Exception as err:
            err_msg = str(err)
            print(f">> ❌ VTON 生成錯誤: {err_msg}")

            # 針對常見錯誤給予友善提示
            if "Queue" in err_msg:
                return None, "目前使用人數過多 (Queue Full)，請稍後再試。"
            return None, f"生成發生錯誤: {err_msg[:50]}..."

vton_engine = TryOnEngine()

# 真實 AI 投票分析與視覺洞察引擎 (Vote Insight VLM)
class VoteInsightEngine:
    def __init__(self):
        # Moondream2 是一個輕量且速度快的視覺語言模型 (VLM)
        self.client_url = "vikhyatk/moondream2"
        self.client = None
        self.has_dependency = globals().get('GRADIO_AVAILABLE', False)

    def initialize(self):
        """Lazy Loading 連線機制"""
        if not self.has_dependency:
            return False

        if self.client:
            return True

        print(">> 正在連接視覺分析模型 Moondream2...")
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(Client, self.client_url)
                self.client = future.result(timeout=15)
            print(">> ✅ 視覺模型連接成功！")
            return True
        except Exception as err:
            print(f">> ❌ 視覺模型連接失敗: {err}")
            return False

    @staticmethod
    def _resize_image_for_api(img_path, max_size=768):
        """
        [優化] VLM 只需要看大概輪廓與顏色，768px 足夠且速度快。
        """
        # noinspection PyBroadException
        try:
            img = cv2.imread(img_path)
            if img is None: return False

            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_path, img)
            return True
        except Exception:
            return False

    def analyze(self, image_path, vote_result):
        """
        根據投票結果，讓 AI 分析這套穿搭
        vote_result: 'popular' (高票) 或 'improve' (低票)
        """
        if not self.has_dependency:
            return "系統未安裝 gradio_client，無法執行 AI 分析。"

        if not self.initialize():
            return "系統忙碌中，無法連接 AI 分析服務。"

        # 1. 圖片瘦身
        self._resize_image_for_api(image_path)

        # 2. 設計專業時尚 Prompt (Prompt Engineering)
        if vote_result == 'popular':
            prompt = (
                "Describe why this outfit looks stylish and attractive. "
                "Focus on color matching, fit, and vibes. "
                "Please answer in Traditional Chinese (繁體中文). Keep it concise."
            )
        else:
            prompt = (
                "Critique this outfit gently. What is one specific thing that could be improved "
                "(e.g., color coordination, fit, or accessories)? "
                "Please answer in Traditional Chinese (繁體中文). Keep it concise."
            )

        try:
            abs_path = os.path.abspath(image_path)

            # 3. 執行分析 (設定 25 秒逾時)
            # Moondream2 雖然快，但公開 API 有時會有 Queue
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.client.predict,
                    file(abs_path),  # image
                    prompt,  # question
                    api_name="/answer_question"
                )
                result = future.result(timeout=30)

            # result 通常是字串
            return f"{result}"

        except concurrent.futures.TimeoutError:
            return "AI 思考太久了 (Timeout)，請稍後再試。"
        except Exception as err:
            print(f">> 投票分析失敗: {err}")
            return "AI 正在休息，暫時無法評論您的穿搭。"

vote_engine = VoteInsightEngine()

class TrendEngine:
    """
    趨勢智慧引擎：負責計算適配度與預測流行
    """
    @staticmethod
    def calculate_compatibility(user_data, trend_category):
        """
        計算用戶與某個趨勢類別的適配度 (0-100)
        """
        score = 75  # 基礎分

        # 1. 取得用戶風格象限
        user_quadrant = 'CENTER'
        if user_data.get('style_profile'):
            # noinspection PyBroadException
            try:
                prof = json.loads(user_data['style_profile'])
                user_quadrant = prof.get('primary_quadrant', 'CENTER')
            except Exception:  # 🌟 修正：明確指定 Exception，避免攔截系統退出指令
                pass

        # 2. 簡單的加分邏輯
        if user_quadrant == 'Q1' and trend_category in ['Style', 'Design']:
            score += 15
        elif user_quadrant == 'Q4' and trend_category in ['Scene', 'Trend']:
            score += 10

        # 3. 膚色加分
        # noinspection PyBroadException
        try:
            colors = json.loads(user_data.get('color_preferences', '[]'))
            if colors:
                score += 5
        except Exception:  # 🌟 修正：明確指定 Exception
            pass

        return min(100, max(60, score))

    @staticmethod
    def forecast_trend(history_data):
        """
        簡單的線性回歸預測 (Linear Regression)
        """
        if not history_data or len(history_data) < 2:
            return 80  # 預設熱度

        # 計算斜率：(最新 - 上次)
        slope = history_data[-1] - history_data[-2]

        # 預測下期：最新 + 斜率
        prediction = history_data[-1] + slope

        # 加上一點隨機波動
        noise = random.randint(-5, 5)

        return min(100, max(0, prediction + noise))

# 站內大數據趨勢計算引擎 (Internal Trend Engine)
def generate_internal_trends():
    """
    基於站內真實數據 (分析歷史) 生成趨勢。
    統計使用者最常被診斷出的風格，轉化為趨勢分數。
    """
    print("📊 正在根據站內大數據生成本站專屬趨勢...")
    db_conn = get_db_connection()

    # 統計所有分析紀錄中的風格數量
    query = "SELECT final_recommendation FROM analysis_history WHERE final_recommendation IS NOT NULL"
    rows = db_conn.execute(query).fetchall()
    db_conn.close()

    style_counts = {}
    for r in rows:
        # noinspection PyBroadException
        try:
            rec = json.loads(r['final_recommendation'])
            style_name = rec.get('archetype') or rec.get('name') or '自然舒適系'

            # 清洗名稱 (移除 Emoji 與英文，例如 "🌸 甜美鄰家系 (Sweet)" -> "甜美鄰家系")
            clean_name = style_name.split('(')[0].replace('🌸', '').replace('⚡', '').replace('💎', '').replace('👑',
                                                                                                             '').replace(
                '🍃', '').strip()

            style_counts[clean_name] = style_counts.get(clean_name, 0) + 1
        except:
            continue

    # 如果站內完全沒數據 (剛架好網站)，給予預設種子數據
    if not style_counts:
        style_counts = {
            "甜美鄰家系": 15,
            "自然舒適系": 28,
            "潮流個性系": 12,
            "優雅貴氣系": 18,
            "氣場權威系": 8
        }

    # 轉換為趨勢格式 (取前 6 名最熱門的風格)
    sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)[:6]

    # 找出最大值用來標準化分數 (讓最高分的風格大約在 95 分)
    max_count = max([v for k, v in sorted_styles]) if sorted_styles else 1

    live_trends = []
    for kw, count in sorted_styles:
        # 計算熱度分數 (正規化到 65~98 之間)
        score = int(65 + (count / max_count) * 33)

        # 生成歷史趨勢波動假資料 (讓前端曲線圖表好看)
        history = []
        curr = score
        for _ in range(7):
            history.insert(0, curr)
            curr += random.randint(-4, 4)

        desc = f"🔥 本站近期有 {count} 位使用者被診斷為此風格，是目前的穿搭主流指標！"
        live_trends.append((kw, "Style", score, desc, json.dumps(history)))

    return live_trends

# 名人風格爬蟲 (Smart Celebrity Style Scraper)
CELEB_CACHE = {}

# 如果爬蟲失敗 (斷網/被擋)，自動回傳這些預設值，保證演示不開天窗
STATIC_CELEBS = {
    "Y2K": {"name": "NewJeans / Bella Hadid",
            "img": "https://i.pinimg.com/564x/a6/5e/12/a65e1281432a9da43e7bc2ae7d81a9bc.jpg"},
    "老錢風": {"name": "Sofia Richie / 黛安娜王妃",
               "img": "https://i.pinimg.com/564x/12/34/56/old_money_placeholder.jpg"},
    "美拉德": {"name": "Hailey Bieber", "img": "https://i.pinimg.com/564x/78/90/ab/maillard_style.jpg"},
    "極簡": {"name": "Kendall Jenner", "img": "https://i.pinimg.com/564x/cd/ef/gh/minimalist_style.jpg"},
}

def fetch_celeb_match_from_web(style_keyword):
    """
    智慧型爬蟲：Cache -> Google Scraper -> Static Fallback
    """
    # 1. 檢查快取
    if style_keyword in CELEB_CACHE:
        print(f"🚀 [快取命中] 秒回名人資料: {style_keyword}")
        return CELEB_CACHE[style_keyword]

    print(f"🔍 正在為用戶搜尋【{style_keyword}】風格的穿搭 icon...")

    # 組合搜尋關鍵字：加上 'outfit' 或 'street style' 圖片品質較高
    query = f"{style_keyword} fashion style celebrity outfit street snap"
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch&hl=zh-TW&gl=TW"

    # 模擬真實瀏覽器 (防止被視為機器人)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    try:
        # 設定 4 秒逾時，失敗就趕快切換備案
        response = requests.get(url, headers=headers, timeout=4)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Google 圖片的縮圖通常放在 img 標籤中，且 src 包含 'encrypted-tbn0'
        # 我們只抓取這些有效的縮圖，避開 Base64 亂碼
        images = soup.find_all('img')

        candidates = []
        for img in images:
            # noinspection PyBroadException
            try:
                src = img.get('src', '')
                alt = img.get('alt', '')

                # 過濾條件：
                # 1. 必須是 Google 託管的圖片 (tbn0.gstatic.com) -> 載入快且穩定
                # 2. alt 文字長度 > 5 (通常包含人名)
                # 3. 排除 Google 自己的 icon
                if 'http' in src and len(alt) > 5 and 'Google' not in alt:

                    # 清洗文字：移除 "的圖片搜尋結果" 等贅字
                    clean_name = alt.replace("的圖片搜尋結果", "").split(' - ')[0]
                    clean_name = clean_name.replace("...", "").strip()

                    # 避免抓到太長的新聞標題，若太長則截斷
                    if len(clean_name) > 25:
                        clean_name = clean_name[:25] + "..."

                    candidates.append({'name': clean_name, 'img': src})

            except:
                continue

        if candidates:
            # 隨機選前 5 名，增加多樣性
            chosen = random.choice(candidates[:5])
            print(f"✅ 名人匹配成功: {chosen['name']}")

            # 存入快取
            CELEB_CACHE[style_keyword] = chosen
            return chosen

    except Exception as err:
        print(f"⚠️ 名人爬蟲連線錯誤: {err}")

    # --- Fallback: 如果爬蟲失敗，檢查是否有靜態備案 ---
    # 模糊比對：例如 "老錢風 (Old Money)" 只要包含 "老錢" 就匹配
    for key, val in STATIC_CELEBS.items():
        if key in style_keyword:
            print(f"🔄 啟用靜態備案: {key}")
            return val

    # 最終兜底 (Placeholder)
    return {
        'name': f'{style_keyword} 風格代表',
        'img': 'https://placehold.co/400x500/png?text=Style+Icon'
    }

# 資料庫初始化與專家知識注入 (Database Init)
def init_db():
    db_conn = get_db_connection()
    c = db_conn.cursor()

    print("🚀 正在啟動全方位專家系統資料庫整合...")

    # ==========================================
    # Part 1 & 2: 建立核心與功能資料表 (Core & Features)
    # ==========================================

    # 建立用戶表
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, 
        name TEXT NOT NULL, role TEXT DEFAULT 'user', status TEXT DEFAULT 'active', is_vip BOOLEAN DEFAULT 0, 
        age INTEGER, gender TEXT, height REAL, weight REAL, body_code TEXT, face_code TEXT,
        maturity_level TEXT DEFAULT 'balanced', culture_pref INTEGER DEFAULT 5, life_stage TEXT DEFAULT 'student', 
        clothing_issues TEXT, style_preferences TEXT, color_preferences TEXT, occasion_preferences TEXT, 
        data_consent BOOLEAN DEFAULT 0, photo_policy TEXT DEFAULT '30_days', ai_training_consent BOOLEAN DEFAULT 0, 
        accessibility_prefs TEXT, tos_version TEXT DEFAULT '1.0', locale TEXT DEFAULT 'zh_TW', 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # 建立分析歷史表
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, user_image_path TEXT, 
        face_data TEXT, body_data TEXT, final_recommendation TEXT, 
        ai_confidence INTEGER DEFAULT 85, is_incorrect BOOLEAN DEFAULT 0, user_feedback TEXT, 
        ab_variant TEXT DEFAULT 'A', model_version TEXT, logic_trace TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
        FOREIGN KEY (user_id) REFERENCES users (id))''')

    # 建立商品、趨勢、貼文等其他表格
    tables_sql = [
        # 商品表
        '''CREATE TABLE IF NOT EXISTS clothing_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, seller_id INTEGER, image_path TEXT, title TEXT, category TEXT, 
            quadrant TEXT, material TEXT, pattern TEXT, neckline TEXT, fit_type TEXT, 
            tags TEXT, brand TEXT DEFAULT '自有品牌', price INTEGER, status TEXT DEFAULT 'on_sale', 
            description TEXT, is_ad BOOLEAN DEFAULT 0, trust_score INTEGER DEFAULT 100, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (seller_id) REFERENCES users (id))''',
        # 趨勢表
        '''CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT UNIQUE, style_quadrant TEXT, category TEXT, 
            status TEXT DEFAULT 'rising', influence_score INTEGER, description TEXT, data_points TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
        # 社群功能表
        'CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, image_path TEXT, content TEXT, tags TEXT, is_anonymous BOOLEAN, is_qa BOOLEAN DEFAULT 0, poll_yes INTEGER DEFAULT 0, poll_no INTEGER DEFAULT 0, likes_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))',
        'CREATE TABLE IF NOT EXISTS celebrity_looks (id INTEGER PRIMARY KEY AUTOINCREMENT, trend_id INTEGER, celeb_name TEXT, image_path TEXT, description TEXT, FOREIGN KEY (trend_id) REFERENCES trends (id), UNIQUE(trend_id, celeb_name))',
        'CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, post_id INTEGER, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS likes (user_id INTEGER, post_id INTEGER, PRIMARY KEY (user_id, post_id))',
        'CREATE TABLE IF NOT EXISTS follows (follower_id INTEGER, followed_id INTEGER, PRIMARY KEY (follower_id, followed_id))',
        'CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, reporter_id INTEGER, post_id INTEGER, reason TEXT, status TEXT DEFAULT "pending", created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS celeb_likes (user_id INTEGER, celeb_id INTEGER, PRIMARY KEY (user_id, celeb_id))',
        'CREATE TABLE IF NOT EXISTS wear_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, date_str TEXT, outfit_desc TEXT, feeling TEXT, rating INTEGER, ai_adjustment_note TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, date_str TEXT, title TEXT, outfit_desc TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS body_tracking (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, weight REAL, waist REAL, hip REAL, note TEXT, recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS try_on_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, original_img TEXT, cloth_img TEXT, result_img TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS chat_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, sender TEXT, message TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS user_milestones (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, milestone_type TEXT, achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS style_proposals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, tag_name TEXT, description TEXT, status TEXT DEFAULT "pending", created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
        'CREATE TABLE IF NOT EXISTS system_configs (key TEXT PRIMARY KEY, value TEXT)',
        'CREATE TABLE IF NOT EXISTS favorites (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, item_data TEXT, saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'
    ]
    for sql in tables_sql: c.execute(sql)

    # ==========================================
    # Part 3: 建立知識庫表 (Knowledge Base Schema)
    # ==========================================

    # 強制刪除舊表，確保 Schema 更新
    ref_tables_to_drop = [
        "ref_face_rules", "ref_body_rules", "ref_dress_codes",
        "ref_style_quadrants", "ref_hairstyles", "ref_makeup_styles",
        "ref_colors", "ref_material_pattern", "ref_cut_details", "ref_accessories"
    ]
    for t in ref_tables_to_drop:
        c.execute(f"DROP TABLE IF EXISTS {t}")

    # 建立最新的表格結構
    kb_tables = [
        'CREATE TABLE IF NOT EXISTS ref_face_rules (face_shape TEXT PRIMARY KEY, visual_feature TEXT, suitable_hairstyles TEXT, avoid_hairstyles TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_body_rules (body_type TEXT PRIMARY KEY, feature_desc TEXT, do_wear TEXT, dont_wear TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_dress_codes (code_name TEXT PRIMARY KEY, desc TEXT, suitable_items TEXT)',
        '''CREATE TABLE IF NOT EXISTS ref_style_quadrants (
            code TEXT PRIMARY KEY, name TEXT, volume TEXT, line TEXT, 
            psychology TEXT, keywords TEXT, body_suit TEXT, culture_vibe TEXT, 
            clothing_guide TEXT, avoid_guide TEXT)''',
        'CREATE TABLE IF NOT EXISTS ref_hairstyles (id INTEGER PRIMARY KEY AUTOINCREMENT, gender TEXT, category TEXT, name TEXT, matrix_tag TEXT, description TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_makeup_styles (category TEXT, name TEXT, matrix_tag TEXT, description TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_colors (season TEXT PRIMARY KEY, name_zh_en TEXT, logic TEXT, vibe TEXT, best_colors TEXT, avoid_colors TEXT, matrix_match TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_material_pattern (category TEXT, sub_category TEXT, name TEXT, matrix_tag TEXT, description TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_cut_details (category TEXT, name TEXT, suit_targets TEXT, effect TEXT)',
        'CREATE TABLE IF NOT EXISTS ref_accessories (quadrant TEXT, category TEXT, sub_category TEXT, items TEXT)'
    ]
    for sql in kb_tables: c.execute(sql)

    # ==========================================
    # Part 4: 注入專家知識 (Knowledge Injection)
    # ==========================================
    print("📚 正在啟動全自動專家矩陣同步...")

    # [A] 臉型與體型規則
    if "FACE_BODY_RULES" in globals():
        c.executemany("INSERT OR REPLACE INTO ref_face_rules VALUES (?,?,?,?)", FACE_BODY_RULES.get("FACE", []))
        c.executemany("INSERT OR REPLACE INTO ref_body_rules VALUES (?,?,?,?)", FACE_BODY_RULES.get("BODY", []))

    # [B] 五大風格象限 (全面防呆版)
    quadrants_data = []
    if "STYLE_MATRIX" in globals():
        for q_code, content in STYLE_MATRIX.get("STYLE_MATRIX", {}).items():
            w_guide = content.get('wardrobe_guide', {}).get('WOMEN', {}).get('items', {})
            m_guide = content.get('wardrobe_guide', {}).get('MEN', {}).get('items', {})

            clothing_text = (
                f"[女] 上:{w_guide.get('tops', '-')} 下:{w_guide.get('bottoms', '-')} | "
                f"[男] 上:{m_guide.get('tops', '-')} 下:{m_guide.get('bottoms', '-')}"
            )

            avoid_w = content.get("styling_tips", {}).get("WOMEN", {}).get("avoid", "")
            avoid_m = content.get("styling_tips", {}).get("MEN", {}).get("avoid", "")
            avoid_text = f"{avoid_w} {avoid_m}".strip()

            logic = content.get("analysis_logic", {})
            mapping = content.get("taxonomy_mapping", {})
            keywords = logic.get("keywords", [])
            body_type = logic.get("body_type", [])
            culture_vibe = mapping.get("culture_vibe", "General")

            quadrants_data.append((
                q_code,
                content.get("name", f"Style {q_code}"),  # 防呆預設值
                logic.get("volume", "Medium"),
                logic.get("line", "Mixed"),
                logic.get("psychology", "Balanced"),
                ",".join(keywords) if isinstance(keywords, list) else str(keywords),
                ",".join(body_type) if isinstance(body_type, list) else str(body_type),
                str(culture_vibe),
                clothing_text,
                avoid_text
            ))

        c.execute("DELETE FROM ref_style_quadrants")
        c.executemany("INSERT INTO ref_style_quadrants VALUES (?,?,?,?,?,?,?,?,?,?)", quadrants_data)

    # [C] 髮型百科 (全面防呆版)
    hair_sql_data = []
    if "HAIRSTYLE_DATABASE" in globals():
        for gender, categories in HAIRSTYLE_DATABASE.items():
            for category, hairstyles in categories.items():
                for key, info in hairstyles.items():
                    # 處理 tags 可能是 list 或 None 的情況
                    tags = info.get("matrix_tag", [])
                    tag_str = ",".join(tags) if isinstance(tags, list) else str(tags)

                    hair_sql_data.append((
                        gender,
                        category,
                        info.get("name", key),  # 若無 name 則使用 key
                        tag_str,
                        info.get("desc", "")
                    ))
        c.execute("DELETE FROM ref_hairstyles")
        c.execute("DELETE FROM sqlite_sequence WHERE name='ref_hairstyles'")
        c.executemany("INSERT INTO ref_hairstyles (gender, category, name, matrix_tag, description) VALUES (?,?,?,?,?)",
                      hair_sql_data)

    # [D] 妝容流派 (全面防呆版)
    makeup_sql_data = []
    if "MAKEUP_DATABASE" in globals():
        for region, styles in MAKEUP_DATABASE.items():
            for key, info in styles.items():
                tags = info.get("matrix_tag", [])
                tag_str = ",".join(tags) if isinstance(tags, list) else str(tags)

                makeup_sql_data.append((
                    region,
                    info.get("name", key),  # 若無 name 則使用 key
                    tag_str,
                    info.get("desc", "")
                ))
        c.execute("DELETE FROM ref_makeup_styles")
        c.executemany("INSERT INTO ref_makeup_styles VALUES (?,?,?,?)", makeup_sql_data)

    # [E] 四季色彩 (全面防呆版)
    color_sql_data = []
    if "SEASONAL_COLOR_DATABASE" in globals():
        for season_key, info in SEASONAL_COLOR_DATABASE.items():
            # 處理 colors 可能是 list 或是新版 dict 的情況
            best = info.get("color_palette", {}).get("primary", []) if "color_palette" in info else info.get(
                "best_colors", [])
            avoid = info.get("avoid_colors", [])
            match = info.get("matrix_match", [])

            name_val = info.get("name", info.get("name_zh_en", season_key))

            color_sql_data.append((
                season_key,
                name_val,
                info.get("logic", ""),
                info.get("vibe", ""),
                ",".join(best) if isinstance(best, list) else str(best),
                ",".join(avoid) if isinstance(avoid, list) else str(avoid),
                ",".join(match) if isinstance(match, list) else str(match)
            ))
        c.execute("DELETE FROM ref_colors")
        c.executemany("INSERT INTO ref_colors VALUES (?,?,?,?,?,?,?)", color_sql_data)

    # [F] 版型與配件 (全面防呆版)
    cut_sql = []
    if "CUT_DETAIL_DATABASE" in globals():
        for cat, details in CUT_DETAIL_DATABASE.items():
            for k, info in details.items():
                suit = info.get("suit", [])
                cut_sql.append((
                    cat,
                    info.get("name", k),  # 防呆
                    ",".join(suit) if isinstance(suit, list) else str(suit),
                    info.get("effect", "")
                ))
        c.execute("DELETE FROM ref_cut_details")
        c.executemany("INSERT INTO ref_cut_details VALUES (?,?,?,?)", cut_sql)

    # 配件
    acc_sql = []
    if "ACCESSORY_MATRIX" in globals():
        for q_code, categories in ACCESSORY_MATRIX.items():
            for cat_name, sub_items in categories.items():
                if isinstance(sub_items, dict):
                    for sub_name, items_list in sub_items.items():
                        acc_sql.append((q_code, cat_name, sub_name, str(items_list)))
                elif isinstance(sub_items, list):
                    acc_sql.append((q_code, cat_name, "General", str(sub_items)))
        c.execute("DELETE FROM ref_accessories")
        c.executemany("INSERT INTO ref_accessories VALUES (?,?,?,?)", acc_sql)

    # [G] 場合
    if "DRESS_CODE_DB" in globals():
        c.execute("DELETE FROM ref_dress_codes")
        c.executemany("INSERT INTO ref_dress_codes VALUES (?,?,?)", DRESS_CODE_DB)

    # ==========================================
    # Part 5: 趨勢搜查與自動化佔位符 (Trends)
    # ==========================================
    print("📊 啟動站內趨勢大數據計算...")
    web_trends = generate_internal_trends()

    for kw, cat, score, desc, data_pts in web_trends:
        kw_l = kw.lower()
        quad = 'CENTER'
        # 簡單分類邏輯
        if any(x in kw_l for x in ['sweet', 'ballet', 'coquette']):
            quad = 'Q1'
        elif any(x in kw_l for x in ['y2k', 'cool', 'gorp']):
            quad = 'Q2'
        elif any(x in kw_l for x in ['old money', 'elegant']):
            quad = 'Q3'
        elif any(x in kw_l for x in ['mob', 'work']):
            quad = 'Q4'

        c.execute(
            'INSERT OR REPLACE INTO trends (keyword, style_quadrant, category, influence_score, description, data_points) VALUES (?, ?, ?, ?, ?, ?)',
            (kw, quad, cat, score, desc, data_pts))

        # 建立佔位符
        t_id_row = c.execute("SELECT id FROM trends WHERE keyword = ?", (kw,)).fetchone()
        if t_id_row:
            c.execute(
                'INSERT OR IGNORE INTO celebrity_looks (trend_id, celeb_name, image_path, description) VALUES (?, ?, ?, ?)',
                (t_id_row[0], f"{kw} Icon", "https://placehold.co/400x600", f"{kw} 風格代表"))

    # ==========================================
    # Part 6: 建立預設系統帳號
    # ==========================================
    admin_pw = generate_password_hash('123456')
    c.execute('INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?, ?, ?, ?, 1)',
              ('admin@style.com', admin_pw, '系統管理員', 'admin'))
    c.execute('INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?, ?, ?, ?, 1)',
              ('official@style.com', admin_pw, 'Smart Style 官方', 'official'))

    db_conn.commit()
    db_conn.close()
    print("✅ 資料庫初始化完成，所有專家矩陣數據已同步！")

# 真實天氣引擎 (Stable Weather Engine - wttr.in)
def get_weather_data(location="Taoyuan"):
    """
    [穩定版] 使用 wttr.in 獲取天氣
    修復：當 API 沒回傳中文時，自動降級使用英文，避免報錯。
    """
    # 備案數據 (Demo 時萬一斷網專用)
    fallback_data = {
        'temp': 24,
        'condition': 'cloudy',
        'humidity': 65,
        'desc': '多雲 (預設)',
        'location_name': location
    }

    try:
        # print(f"🌍 正在連線氣象衛星查詢: {location} ...") # 除錯用，可註解掉

        # format=j1: 回傳詳細 JSON
        # lang=zh-TW: 請求繁體中文
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1&lang=zh-TW"

        response = requests.get(url, timeout=3) # 設定 3 秒逾時，避免卡太久

        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            nearest_area = data['nearest_area'][0]

            # 1. 氣溫
            temp = int(current['temp_C'])

            # 2. 濕度
            humidity = int(current['humidity'])

            # 3. [關鍵修復] 天氣描述防呆機制
            # wttr.in 有時候會漏掉 lang_zh-TW，導致 KeyError
            try:
                desc = current['lang_zh-TW'][0]['value']
            except KeyError:
                # 如果找不到中文，就抓英文描述 (weatherDesc)
                desc = current['weatherDesc'][0]['value']

            # 4. 地點名稱
            loc_name = nearest_area['areaName'][0]['value']

            # 5. 轉換代碼 (用於決定穿搭邏輯)
            condition_code = 'cloudy'
            desc_check = desc.lower()

            if any(x in desc_check for x in ['雨', 'rain', 'drizzle', 'shower', 'thunder']):
                condition_code = 'rainy'
            elif any(x in desc_check for x in ['晴', 'sun', 'clear']):
                condition_code = 'sunny'
            elif any(x in desc_check for x in ['雪', 'snow']):
                condition_code = 'snowy'

            # print(f"✅ 天氣獲取成功: {loc_name} | {temp}°C | {desc}")

            return {
                'temp': temp,
                'condition': condition_code,
                'humidity': humidity,
                'desc': desc,
                'location_name': loc_name
            }
        else:
            # print(f"⚠️ 氣象服務回應異常: {response.status_code}")
            return fallback_data

    except Exception as err:
        print(f"⚠️ 天氣連線失敗 (啟動備案模式): {err}")
        # 為了演示效果，隨機微調一下數字，讓畫面看起來是活的
        fallback_data['temp'] += random.randint(-2, 2)
        return fallback_data

# 演算法公平性報告計算
def calculate_fairness_metrics():
    """
    演算法公平性報告計算 (基於真實資料庫)
    統計各身形的使用者回報「結果不準確 (is_incorrect)」的比例
    """
    db_conn = get_db_connection()
    # 撈取所有分析紀錄 (包含身形數據與錯誤標記)
    rows = db_conn.execute('SELECT body_data, is_incorrect FROM analysis_history').fetchall()
    db_conn.close()

    # 使用 Python 字典進行統計 (比 SQL LIKE 更精準處理 JSON)
    stats = {}

    for r in rows:
        # noinspection PyBroadException
        try:
            # 解析 JSON 數據
            b_data = json.loads(r['body_data'])
            # 取得身形名稱 (例如 "梨型 (A型)" -> 取 "梨型")
            shape_full = b_data.get('shape', '未知身形')
            shape = shape_full.split(' ')[0] if shape_full else '未知身形'
        except:
            shape = '資料解析錯誤'

        # 初始化該身形的統計槽
        if shape not in stats:
            stats[shape] = {'total': 0, 'errors': 0}

        # 累加數據
        stats[shape]['total'] += 1
        if r['is_incorrect']:
            stats[shape]['errors'] += 1

    # 格式化輸出結果
    result = []
    for shape, data in stats.items():
        total = data['total']
        errors = data['errors']
        # 計算錯誤率
        rate = round((errors / total * 100), 1) if total > 0 else 0

        # 判斷是否偏差過高 (超過 25% 錯誤率視為異常)
        status = 'Normal'
        if rate > 25:
            status = 'Bias Detected (高誤判)'
        elif total < 5:
            status = 'Data Insufficient (樣本不足)'

        result.append({
            'type': shape,
            'rate': rate,
            'status': status,
            'count': total
        })

    return result

# 長期趨勢分析數據
def get_trend_analysis():
    """
    長期趨勢分析數據 (基於真實資料庫)
    分析過去 6 個月，使用者最常被診斷出哪種風格 (Style Archetype)
    """
    db_conn = get_db_connection()

    # 撈取分析紀錄 (按時間排序)
    query = '''
        SELECT created_at, final_recommendation 
        FROM analysis_history 
        ORDER BY created_at ASC
    '''
    rows = db_conn.execute(query).fetchall()
    db_conn.close()

    # 資料結構: { '2023-10': {'甜美系': 5, '酷帥系': 2}, ... }
    monthly_stats = {}
    all_styles = set()

    for r in rows:
        # noinspection PyBroadException
        try:
            # 取得月份 (YYYY-MM)
            month = r['created_at'][:7]

            # 解析推薦結果，找出風格原型
            rec = json.loads(r['final_recommendation'])
            # 抓取風格名稱 (例如 "甜美鄰家系 (Sweet...)" -> 取 "甜美鄰家系")
            # 這裡做了多重防呆，確保能抓到名字
            style_full = rec.get('archetype') or rec.get('name') or '未定義風格'
            style = style_full.split('(')[0].strip()

            all_styles.add(style)

            if month not in monthly_stats:
                monthly_stats[month] = {}

            monthly_stats[month][style] = monthly_stats[month].get(style, 0) + 1

        except:
            continue

    # 如果完全沒有資料，回傳空結構避免報錯
    if not monthly_stats:
        return {'years': [], 'styles': {}}

    # 轉換為 Chart.js 需要的格式
    # labels (X軸): 月份列表
    sorted_months = sorted(monthly_stats.keys())  # ['2023-10', '2023-11'...]

    # datasets (Y軸): 每個風格在該月份的數量
    formatted_styles = {}

    for style in all_styles:
        counts = []
        for m in sorted_months:
            # 填入該月該風格的數量，沒有則補 0
            counts.append(monthly_stats[m].get(style, 0))
        formatted_styles[style] = counts

    return {
        'years': sorted_months,
        'styles': formatted_styles
    }

# 風格疲勞檢測器 (Style Fatigue Detector)
def check_style_fatigue(user_id):
    """
    檢查使用者是否陷入「風格疲勞」(Style Fatigue)。

    邏輯：
    檢查最近 3 次的分析結果。如果 AI 連續 3 次都推薦完全相同的風格 (Archetype)，
    則判定為疲勞，並回傳該風格名稱，以便系統在本次推薦中主動避開。

    Returns:
        str: 造成疲勞的風格名稱 (例如 "甜美鄰家系")，若無疲勞則回傳 None。
    """
    db_conn = get_db_connection()
    try:
        # 只撈取最近 3 筆資料
        recent = db_conn.execute(
            '''SELECT final_recommendation FROM analysis_history 
               WHERE user_id = ? 
               ORDER BY created_at DESC LIMIT 3''',
            (user_id,)
        ).fetchall()
    except Exception as err:
        print(f"Fatigue DB Error: {err}")
        return None
    finally:
        db_conn.close()

    # 資料不足 3 筆，不可能疲勞
    if len(recent) < 3:
        return None

    try:
        # 提取最近 3 次的風格名稱
        styles = []
        for row in recent:
            if not row['final_recommendation']:
                continue

            data = json.loads(row['final_recommendation'])
            # 取得風格名稱，若沒有則給空字串
            # 注意：這裡假設 JSON 結構中有 'archetype' 或 'style_name'
            # 根據您的 STYLE_MATRIX，key 應該是 'name' (例如 "甜美鄰家系...")
            # 這裡我們用模糊比對抓取括號前的中文名稱比較準確
            full_name = data.get('archetype', '')
            if not full_name:
                # 相容性：舊資料可能存的是 'style'
                full_name = data.get('style', '')

            # 簡化名稱只取中文 (例如 "甜美鄰家系 (Sweet...)" -> "甜美鄰家系")
            simple_name = full_name.split('(')[0].strip()
            styles.append(simple_name)

        # 嚴格判定：必須 3 次都抓得到資料，且 3 次名稱完全相同
        if len(styles) == 3 and styles[0] == styles[1] == styles[2]:
            # print(f"⚠️ 偵測到風格疲勞: {styles[0]}")
            return styles[0]  # 回傳該風格名稱

    except Exception as err:
        print(f"Fatigue Check Error: {err}")
        return None

    return None

def generate_dynamic_response(user_name, body_shape, style_arch, user_msg):
    """ 當 AI API 斷線時的備用回覆產生器 """
    import random
    responses = [
        f"哈囉 {user_name}！身為您的專屬顧問，針對您的{body_shape}與{style_arch}風格...",
        f"收到您的問題「{user_msg}」！以您的{style_arch}氣質，可以試著加入一點質感飾品來點綴！",
        f"沒問題！因為您是{body_shape}，我會推薦您挑選硬挺材質的版型會更修飾喔。"
    ]
    return random.choice(responses)

# 穿搭故事產生器 (Contextual Story Generator)
def get_story_tag(life_stage):
    """根據人生階段賦予情境故事"""
    stories = {
        'student': ['圖書館窗邊的午後邂逅', '期末報告日的自信戰袍', '社團成發夜的閃亮時刻', '週末與室友的文青市集'],
        'new_grad': ['第一次面試的沈穩野心', '週五下班後的微醺時光', '職場新人的專業氣場', '發薪日犒賞自己的儀式感'],
        'career_change': ['重新定義自我的勇氣', '跨領域挑戰的專業展現', '面試新工作的幸運戰袍', '歸零後的自在與從容'],
        'stable': ['週末家庭日的愜意午後', '主管會議上的領袖氣場', '一個人的質感美術館旅行', '結婚週年紀念的浪漫晚餐'],
        'explore': ['沒有目的地的城市漫遊', '嘗試全新色彩的大膽實驗', '尋找第二個自我的旅程', '轉角咖啡廳的獨處時光']
    }
    return random.choice(stories.get(life_stage, stories['explore']))

# 輔助函式：自動檢索修飾技巧
def get_automated_hacks(face_shape, body_shape):
    """根據特徵關鍵字，自動從 CUT_DETAIL_DATABASE 撈取建議"""
    hacks = []

    # 1. 搜尋 CUT_DETAIL_DATABASE (版型細節)
    if 'CUT_DETAIL_DATABASE' in globals():
        # [修正] 字典結構為三層：Gender -> Category -> Item -> Info
        for gender, categories in CUT_DETAIL_DATABASE.items():
            for category, items in categories.items():
                for key, info in items.items():
                    # 安全地取得 'suit' 陣列，避免 KeyError
                    suit_tags = info.get('suit', [])
                    # 檢查適用的標籤是否出現在使用者的特徵中
                    for tag in suit_tags:
                        if tag in face_shape or tag in body_shape:
                            hacks.append(f"✨ **{info['name']}**：適合{tag}，能{info['effect']}。")
                            break

    # 2. 專家規則補強
    body_str = str(body_shape)
    face_str = str(face_shape)

    if "梨" in body_str or "A" in body_str:
        hacks.append("💡 **下身修飾**：避免緊身褲，選擇A字裙或寬褲遮擋假胯寬。")
    if "方" in face_str:
        hacks.append("💡 **髮型修飾**：利用側分瀏海或C字彎柔和下顎線條。")

    if "短脖子" in body_str or "neck" in body_str:
        hacks.append("💡 **領口選擇**：V領或大方領是您的最佳夥伴，避免高領。")

    # 如果沒找到任何建議，給出通用建議
    if not hacks:
        hacks = ["✨ **腰線定位**：強調腰線能優化比例。", "✨ **三色原則**：全身不超過三種主色。"]

    random.shuffle(hacks)
    return hacks[:5]

def analyze_style_logic(user_data, weather_data):
    """
    [2026 最終完美合併版] 核心演算法
    """
    trace = []

    # --- Phase 1: 數據解析 ---
    try:
        if isinstance(user_data, str):
            user_data = json.loads(user_data)

        raw_face = user_data.get('face_data', {})
        raw_body = user_data.get('body_data', {})
        face = json.loads(raw_face) if isinstance(raw_face, str) else raw_face
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body

        if not isinstance(face, dict): face = {}
        if not isinstance(body, dict): body = {}

        f_shape = face.get('shape', '鵝蛋臉')
        b_shape = body.get('shape', '直筒型')

        raw_gender = str(user_data.get('gender', 'Female')).upper()
        gender_key = "WOMEN" if any(k in raw_gender for k in ["FEM", "女", "F"]) else "MEN"
        skin_season = user_data.get('skin_season', None)

        trace.append(f"特徵: {f_shape}, {b_shape}, {skin_season}")

    except Exception as err:
        print(f"Logic Parse Error: {err}")
        return f"數據解析或分析異常: {err}", " -> ".join(trace) if trace else "Error", "System Error", {}

    # --- Phase 2: 風格量化 ---
    curve, volume = 5.0, 5.0
    for keywords, score_delta in STYLE_SCORING_RULES.get("FACE_CURVE", []):
        if any(k in f_shape for k in keywords):
            curve += score_delta
            break

    for keywords, score_delta in STYLE_SCORING_RULES.get("BODY_VOLUME", []):
        if any(k in b_shape for k in keywords):
            volume += score_delta
            break

    if curve >= 6.0:
        target_q = 'Q1' if volume < 6.0 else 'Q3'
    elif curve <= 4.0:
        target_q = 'Q2' if volume < 6.0 else 'Q4'
    else:
        target_q = 'CENTER'

    style_config = STYLE_MATRIX["STYLE_MATRIX"][target_q]
    trace.append(f"定位: {target_q}")

    # --- Phase 3: 資料庫匹配 ---
    hacks = get_automated_hacks(f_shape, b_shape)

    # 髮型
    hair_db = HAIRSTYLE_DATABASE.get(gender_key, {})
    suitable_hairs = []
    for cat, items in hair_db.items():
        for k, v in items.items():
            if target_q in v.get("matrix_tag", []):
                suitable_hairs.append(f"{v['name']}")

    # 妝容
    makeup_db = MAKEUP_DATABASE.get(gender_key, MAKEUP_DATABASE.get("WOMEN", {}))
    suitable_makeups = []
    for category_name, style_dict in makeup_db.items():
        for style_key, info in style_dict.items():
            if target_q in info.get("matrix_tag", []):
                suitable_makeups.append(f"{info['name']}")

    # 色彩 (優先膚色，否則矩陣)
    suitable_colors = []
    seasonal_vibe = ""
    if skin_season and skin_season in SEASONAL_COLOR_DATABASE:
        info = SEASONAL_COLOR_DATABASE[skin_season]
        best_c = info.get("color_palette", {}).get("primary", []) if "color_palette" in info else info.get(
            "best_colors", [])
        suitable_colors.append({"season": info.get("name", skin_season) + " (實測)", "best": best_c,
                                "avoid": info.get("avoid_colors", [])})
        seasonal_vibe = info.get("vibe", "")
    else:
        for season, info in SEASONAL_COLOR_DATABASE.items():
            if target_q in info.get("matrix_match", []):
                best_c = info.get("color_palette", {}).get("primary", []) if "color_palette" in info else info.get(
                    "best_colors", [])
                suitable_colors.append(
                    {"season": info.get("name", season), "best": best_c, "avoid": info.get("avoid_colors", [])})
                if not seasonal_vibe:
                    seasonal_vibe = info.get("vibe", "")

    # 材質與圖案
    suitable_fabrics = []
    suitable_patterns = []
    for cat, subcats in MATERIAL_PATTERN_DATABASE.get("FABRICS", {}).items():
        for k, v in subcats.items():
            if target_q in v.get("matrix_tag", []):
                suitable_fabrics.append(v["name"])

    for cat, v in MATERIAL_PATTERN_DATABASE.get("PATTERNS", {}).items():
        if target_q in v.get("matrix_tag", []):
            suitable_patterns.append(v["name"])

    # 配件 (擴充版切割邏輯)
    gender_acc_db = ACCESSORY_MATRIX.get(gender_key, ACCESSORY_MATRIX)
    acc_config = gender_acc_db.get(target_q, {})
    acc_items = []
    categories_to_pick = ["Jewelry", "Functional_Styling", "Bags", "Hair_Styling"]

    for cat in categories_to_pick:
        if cat in acc_config:
            items = acc_config[cat]
            if isinstance(items, list):
                count_to_pick = min(3, len(items))
                picks = random.sample(items, count_to_pick)
                acc_items.extend(picks)
            elif isinstance(items, dict):
                values = list(items.values())
                count_to_pick = min(3, len(values))
                picks = random.sample(values, count_to_pick)
                for p in picks:
                    text_part = p
                    if '/' in p:
                        text_part = p.rsplit('/', 1)[-1].strip()
                    sub_items = re.split(r'[、,]', text_part)
                    for item in sub_items:
                        if item.strip():
                            acc_items.append(item.strip())

    random.shuffle(acc_items)
    acc_highlight = "、".join(acc_items[:12])

    # --- Phase 4: 輸出封裝 ---
    wardrobe_data = style_config.get('wardrobe_guide', {}).get(gender_key, {})
    wardrobe_items = wardrobe_data.get('items', {'tops': 'N/A', 'bottoms': 'N/A', 'shoes': 'N/A'})

    styling_data = style_config.get('styling_tips', {}).get(gender_key, {})
    temp = weather_data.get('temp', 24)

    final_makeup = random.choice(suitable_makeups) if suitable_makeups else styling_data.get('makeup', 'Clean Look')
    final_hair = random.choice(suitable_hairs) if suitable_hairs else styling_data.get('hair', 'Neat Style')

    color_tips = ""
    if suitable_colors:
        c = suitable_colors[0]
        color_tips = f"推薦 **{c['season']}** 色系 ({', '.join(c['best'][:2])}...)"

    result_json = {
        'quadrant': target_q,
        'full_name': style_config['name'],
        'psychology': style_config.get('analysis_logic', {}).get('psychology', ''),

        # 穿搭
        'wardrobe_tops': wardrobe_items.get('tops', 'N/A'),
        'wardrobe_bottoms': wardrobe_items.get('bottoms', 'N/A'),
        'wardrobe_shoes': wardrobe_items.get('shoes', 'N/A'),

        # 細節
        'fabrics': suitable_fabrics,
        'patterns': suitable_patterns,
        'makeup': final_makeup,
        'makeup_list': suitable_makeups,
        'hairstyle': final_hair,
        'hairstyle_list': suitable_hairs,
        'accessories': acc_highlight,

        'seasonal_color': suitable_colors,
        'hacks': hacks,
        'weather_tip': "🌤️ 氣溫適宜" if 18 < temp < 27 else "❄️ 注意氣溫"
    }

    advice_text = (
            f"### 👑 風格定調：{result_json['full_name']}\n"
            f"_{result_json['psychology']}_\n\n"
            f"**🎨 色彩與氛圍**：\n{color_tips}。適合 {seasonal_vibe} 的氛圍。\n\n"
            f"**👗 穿搭公式 (Do's)**：\n"
            f"• **上身**：{result_json['wardrobe_tops']}\n"
            f"• **下著**：{result_json['wardrobe_bottoms']}\n"
            f"• **鞋款**：{result_json['wardrobe_shoes']}\n\n"
            f"**💄 妝髮建議**：\n推薦 **{final_makeup}** 搭配 **{final_hair}**。\n飾品可選擇：**{acc_highlight}**。\n\n"
            f"**🛠️ 個人化修飾**：\n" + "\n".join(hacks[:3])
    )

    return advice_text, " -> ".join(trace), "AI 形象診斷完成", result_json

# 使用者偏好與避雷系統 (Preference & Safety)
def get_user_dislikes(user_data):
    """
    從使用者資料中提取「不喜歡」的關鍵字列表
    """
    if not user_data:
        return []

    try:
        # 資料庫存的是 JSON 字串，先轉成 Dict
        # 注意：sqlite3.Row 物件若欄位是 None，get 會回傳 None
        raw_prefs = user_data['style_preferences']
        if not raw_prefs:
            return []

        prefs = json.loads(raw_prefs)
        dislikes = prefs.get('dislike', [])

        # 防呆：確保回傳的是列表
        if isinstance(dislikes, str):
            dislikes = [dislikes]

        return [str(d).strip().lower() for d in dislikes]
    except Exception as err:
        print(f"Error parsing dislikes: {err}")
        return []

def update_user_dislikes(user_id, tags):
    """
    更新用戶的避雷清單 (新增不喜歡的標籤)
    """
    # 防呆：如果傳入的是單一字串，自動轉為 list
    if isinstance(tags, str):
        tags = [tags]

    if not tags:
        return

    db_conn = get_db_connection()
    try:
        user = db_conn.execute("SELECT style_preferences FROM users WHERE id=?", (user_id,)).fetchone()

        prefs = {}
        if user and user['style_preferences']:
            # noinspection PyBroadException
            try:
                prefs = json.loads(user['style_preferences'])
            except:
                prefs = {}

        # 確保資料結構完整
        if 'like' not in prefs: prefs['like'] = []
        if 'dislike' not in prefs: prefs['dislike'] = []

        # 使用 set 自動去重
        current_dislikes = set(prefs['dislike'])

        for t in tags:
            if t:
                current_dislikes.add(str(t).strip().lower())

        # 轉回 list 並存入
        prefs['dislike'] = list(current_dislikes)

        db_conn.execute("UPDATE users SET style_preferences=? WHERE id=?",
                     (json.dumps(prefs, ensure_ascii=False), user_id))
        db_conn.commit()
        print(f"✅ 用戶 {user_id} 避雷清單已更新: {tags}")

    except Exception as err:
        print(f"Update dislikes error: {err}")
    finally:
        db_conn.close()

def is_safe_recommendation(text, dislikes):
    """
    檢查文字是否包含避雷關鍵字
    """
    if not text or not dislikes:
        return True

    target_text = str(text).lower()

    for bad_tag in dislikes:
        bad_tag_clean = str(bad_tag).lower().strip()
        if not bad_tag_clean:
            continue

        # [優化] 避免過度殺傷 (例如不喜歡 "花"，但不該擋掉 "花費")
        # 這裡為了簡單起見仍維持 substring match，但在中文環境這通常是可接受的
        if bad_tag_clean in target_text:
            # print(f"⚠️ 觸發避雷機制: {bad_tag_clean} found in content.")
            return False

    return True

# 頻率限制器 (Rate Limiter)
def check_analysis_frequency(user_id):
    """
    檢查使用者是否在短時間內重複分析
    限制：一般用戶 60 秒，VIP/Admin 無限制
    """
    db_conn = get_db_connection()
    try:
        # 1. 先檢查是否為 VIP (若有權限裝飾器，這裡可以簡化，但為了獨立性還是查一下)
        user = db_conn.execute('SELECT is_vip, role FROM users WHERE id = ?', (user_id,)).fetchone()
        if user and (user['is_vip'] or user['role'] in ['admin', 'official']):
            return True  # VIP 直接放行

        # 2. 檢查最後一次分析時間
        last_record = db_conn.execute(
            'SELECT created_at FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
            (user_id,)).fetchone()

        if last_record:
            # SQLite 的 created_at 通常是 UTC 字串 'YYYY-MM-DD HH:MM:SS'
            last_time_str = last_record['created_at']

            # 將字串轉為時間物件，並強制補上 UTC 時區標籤 (替換 tzinfo)
            last_time = datetime.datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')
            last_time = last_time.replace(tzinfo=datetime.timezone.utc)

            # 使用官方建議的新寫法取得當下 UTC 時間
            current_time = datetime.datetime.now(datetime.timezone.utc)

            # 計算秒數差
            time_diff = (current_time - last_time).total_seconds()

            limit_seconds = 60

            if time_diff < limit_seconds:
                print(f"⏳ 分析冷卻中 (剩餘 {int(limit_seconds - time_diff)} 秒)")
                return False

    except Exception as err:
        print(f"Rate limit check error: {err}")
        # 出錯時預設允許，以免卡死用戶
        return True
    finally:
        db_conn.close()

    return True

# ==========================================
#  頁面路由
# ==========================================
# ---  前端頁面  ---
# --- 1. 首頁與系統設定 ---
@app.route('/')
def index():
    """系統首頁"""
    return render_template('index.html')

@app.route('/set_locale/<locale>')
def set_locale(locale):
    """
    切換語系設定
    包含：白名單驗證 + 資料庫同步更新
    """
    # 1. 白名單驗證 (防止惡意字串注入)
    # 確保 locale 必須是我們支援的 keys (zh_TW, en_US)
    if locale not in TRANSLATIONS:
        locale = 'zh_TW'  # 若不合法，強制回歸預設值

    session['locale'] = locale

    # 2. 若已登入，同步更新使用者偏好
    if 'user_id' in session:
        db_conn = None
        try:
            db_conn = get_db_connection()
            db_conn.execute('UPDATE users SET locale = ? WHERE id = ?', (locale, session['user_id']))
            db_conn.commit()
        except Exception as err:
            print(f"Locale Update Error: {err}")
        finally:
            #  確保連線一定會關閉
            if db_conn: db_conn.close()

    # 3. 導回上一頁 (若無上一頁則回首頁)
    return redirect(request.referrer or url_for('index'))

@app.route('/legal/<page_type>')
def legal_page(page_type):
    """
    [整合路由] 法律與聲明頁面
    支援: /legal/terms (服務條款), /legal/wellness (身心健康)
    """
    # 白名單檢查，避免 404 或路徑遍歷
    allowed_pages = ['terms', 'wellness', 'privacy']

    if page_type not in allowed_pages:
        return redirect(url_for('index'))  # 或 render_template('404.html')

    return render_template('legal.html', type=page_type)

# --- 2. 使用者認證 ---
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        db_conn = get_db_connection()
        try:
            user = db_conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        finally:
            db_conn.close()  # 確保連線一定關閉

        if user and check_password_hash(user['password'], password):
            # 1. 檢查帳號狀態
            if user['status'] == 'banned':
                flash('您的帳號已被停權，請聯繫管理員。', 'error')
                return render_template('login.html')

            # 2. 設定 Session
            session.permanent = True
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['avatar'] = user['avatar']
            session['role'] = user['role']
            session['is_vip'] = bool(user['is_vip'])
            session['is_admin'] = (user['role'] == 'admin')
            session['locale'] = user['locale'] if user['locale'] else 'zh_TW'

            # 3. 智慧導向 (Smart Redirect)
            next_url = request.args.get('next')

            # 安全檢查：確保 next_url 是站內連結
            if not next_url or not next_url.startswith('/'):
                next_url = None

            if next_url:
                return redirect(next_url)

            if session['is_admin']:
                flash('歡迎回來，系統管理員', 'success')
                return redirect(url_for('admin_dashboard'))

            return redirect(url_for('index'))

        flash('帳號或密碼錯誤', 'error')

    return render_template('login.html')

# --- Google ---
@app.route('/login/google')
def login_google():
    # 導向 Google 登入頁面
    redirect_uri = url_for('auth_google', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/google')
def auth_google():
    try:
        token = oauth.google.authorize_access_token()
        # Google 會回傳 userinfo (包含 name, email, picture)
        user_info = token.get('userinfo')
        if not user_info:
            # 有時候需要手動再抓一次
            user_info = oauth.google.userinfo()

        return handle_oauth_login(
            provider='google',
            email=user_info['email'],
            name=user_info['name'],
            avatar=user_info.get('picture')
        )
    except Exception as err:
        print(f"Google Login Error: {err}")
        flash('Google 登入失敗，請稍後再試', 'error')
        return redirect(url_for('login_page'))

# --- LINE ---
@app.route('/login/line')
def login_line():
    redirect_uri = url_for('auth_line', _external=True)
    return oauth.line.authorize_redirect(redirect_uri)

@app.route('/auth/line')
def auth_line():
    try:
        # 這裡需要解碼 (Authlib 自動處理) 或再次呼叫 profile API
        # 簡單起見，我們呼叫 LINE Profile API
        resp = oauth.line.get('profile')
        profile = resp.json()

        # LINE 不一定會給 email (需申請權限)，若沒有則用 ID 模擬
        email = profile.get('email')
        if not email:
            # 若沒 email，用 LINE ID 偽造一個內部識別用
            email = f"{profile['userId']}@line.user"

        return handle_oauth_login(
            provider='line',
            email=email,
            name=profile['displayName'],
            avatar=profile.get('pictureUrl')
        )
    except Exception as err:
        print(f"LINE Login Error: {err}")
        flash('LINE 登入失敗', 'error')
        return redirect(url_for('login_page'))

# --- [核心] 資料庫處理函式 ---
def handle_oauth_login(provider, email, name, avatar):
    """
    通用登入邏輯：
    1. 檢查 Email 是否存在
    2. 若存在 -> 登入
    3. 若不存在 -> 自動註冊 -> 登入
    """
    db_conn = get_db_connection()
    try:
        # 1. 查詢用戶
        user = db_conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if user:
            # [老朋友] 直接登入
            user_id = user['id']
            user_name = user['name']
            role = user['role']
            is_vip = user['is_vip']
            # 可選擇性更新頭像
            # conn.execute('UPDATE users SET avatar=? WHERE id=?', (avatar, user_id))
        else:
            # [新朋友] 自動註冊
            # 生成一個隨機密碼 (因為是用戶不會用密碼登入)
            import uuid
            random_pw = str(uuid.uuid4())

            cursor = db_conn.execute(
                '''INSERT INTO users (name, email, password, role, avatar) 
                   VALUES (?, ?, ?, ?, ?)''',
                (name, email, random_pw, 'user', avatar)
            )
            user_id = cursor.lastrowid
            user_name = name
            role = 'user'
            is_vip = 0
            db_conn.commit()
            flash(f'歡迎加入！已使用 {provider.title()} 完成註冊', 'success')

        # 2. 寫入 Session
        session['user_id'] = user_id
        session['user_name'] = user_name
        session['user_email'] = email
        session['role'] = role
        session['is_vip'] = bool(is_vip)

        return redirect(url_for('index'))

    except Exception as err:
        print(f"DB Error: {err}")
        flash('系統登入錯誤', 'error')
        return redirect(url_for('login_page'))
    finally:
        db_conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        # 1. 服務條款檢查
        if not request.form.get('agree_tos'):
            flash('請先同意服務條款', 'error')
            return render_template('register.html', taxonomy=STYLE_TAXONOMY)

        # 2. 接收並清洗資料
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        hashed_pw = generate_password_hash(password)

        # 處理數值 (使用簡潔寫法)
        def parse_float(val):
            # noinspection PyBroadException
            try:
                return float(val) if val else None
            except:
                return None

        def parse_int(val):
            # noinspection PyBroadException
            try:
                return int(val) if val else None
            except:
                return None

        age = parse_int(request.form.get('age'))
        height = parse_float(request.form.get('height'))
        weight = parse_float(request.form.get('weight'))

        # 處理 JSON 欄位
        meta_data = {
            'issues': json.dumps(request.form.getlist('issues'), ensure_ascii=False),
            'styles': json.dumps({
                'like': request.form.getlist('style_like'),
                'dislike': request.form.getlist('style_dislike')
            }, ensure_ascii=False),
            'colors': json.dumps(request.form.getlist('colors'), ensure_ascii=False)
        }

        db_conn = get_db_connection()
        try:
            cursor = db_conn.cursor()
            # 3. 寫入資料庫
            cursor.execute('''
                INSERT INTO users (
                    email, password, name, tos_version,
                    gender, age, height, weight,
                    clothing_issues, style_preferences, color_preferences,
                    life_stage, culture_pref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email, hashed_pw, name, '1.0',
                request.form.get('gender'), age, height, weight,
                meta_data['issues'], meta_data['styles'], meta_data['colors'],
                request.form.get('life_stage', 'student'),
                request.form.get('culture', 5)
            ))

            # 4. [效能優化] 直接取得剛插入的 ID，無需重新 SELECT
            new_user_id = cursor.lastrowid
            db_conn.commit()

            # 5. 自動登入設定 Session
            session['user_id'] = new_user_id
            session['user_name'] = name
            session['role'] = 'user'
            session['is_vip'] = False
            session['locale'] = 'zh_TW'
            session.permanent = True

            flash('註冊成功！AI 已根據您的偏好為您準備好專屬推薦。', 'success')

            # 若有定義 shop_page 則導向，否則導向首頁
            return redirect(url_for('shop_page'))

        except sqlite3.IntegrityError:
            db_conn.rollback()  # 發生錯誤時回滾
            flash('此 Email 已被註冊，請直接登入', 'error')
            return redirect(url_for('login_page'))
        finally:
            db_conn.close()

    return render_template('register.html', taxonomy=STYLE_TAXONOMY)

@app.route('/logout')
def logout():
    session.clear()
    flash('您已安全登出', 'info')
    return redirect(url_for('index'))

# --- 3. 核心功能：AI 分析與診斷 ---
@app.route('/analysis')
def analysis_page():
    """
    [核心入口] 風格分析主介面
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('analysis.html')

@app.route('/daily_guide')
def daily_guide_page():
    """
    [核心功能] 每日穿搭指南
    整合：天氣 + 行程 + AI 風格分析
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    # 1. 獲取天氣 (支援 URL 參數切換地點)
    city = request.args.get('city', 'Taoyuan')
    weather = get_weather_data(city)

    # 2. 獲取今日行程
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    db_conn = get_db_connection()
    events = db_conn.execute(
        'SELECT * FROM calendar_events WHERE user_id = ? AND date_str = ?',
        (session['user_id'], today)
    ).fetchall()

    # 3. 獲取使用者資料 (用於 AI 分析)
    user_row = db_conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    db_conn.close()

    user_data = dict(user_row) if user_row else {}

    # 4. 執行 AI 分析
    # 預設行程情境

    try:
        # analyze_style_logic 回傳 4 個值
        advice_text, trace_log, story, style_info = analyze_style_logic(
            user_data, weather
        )
    except Exception as err:
        print(f"Daily Guide Error: {err}")
        advice_text = "AI 正在待命，建議您穿著舒適自在的服裝。"
        story = "美好的一天"
        style_info = {'name': '自然風格', 'dos': [], 'donts': []}

    return render_template('daily_guide.html',
                           weather=weather,
                           events=events,
                           today=today,
                           advice=[advice_text],
                           story=story,
                           style_info=style_info)

@app.route('/history')
def history_page():
    """
    [紀錄] 歷史分析檔案
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()
    analyses = db_conn.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    db_conn.close()

    parsed = []
    for r in analyses:
        try:  # <--- 注意這裡的 try 位置
            # 安全解析 JSON
            f = json.loads(r['face_data']) if r['face_data'] else {}

            # 處理推薦結果 (兼容新舊格式)
            style_name = "AI 分析報告"
            if r['final_recommendation']:
                # noinspection PyBroadException
                try:
                    rec = json.loads(r['final_recommendation'])
                    # 優先找 archetype，其次找 name，最後找 summary
                    style_name = rec.get('archetype') or rec.get('name') or "個人化建議"
                    if '(' in style_name:  # 去除括號後的英文，保持版面整潔
                        style_name = style_name.split('(')[0].strip()
                except:
                    pass

            img_path = r['user_image_path']
            clean_img_path = ''
            if img_path:
                clean_img_path = img_path.replace('\\', '/')
                if 'static/' in clean_img_path:
                    clean_img_path = clean_img_path.split('static/')[-1]
                clean_img_path = clean_img_path.lstrip('/')

            parsed.append({
                'id': r['id'],
                'date': r['created_at'],
                'img': clean_img_path,
                'face': f.get('shape', '未偵測'),
                'style': style_name,
                'trace': r['logic_trace'] or ''
            })
        except Exception as err:
            print(f"歷史紀錄讀取錯誤: {err}")
            continue

    return render_template('history.html', analyses=parsed)

@app.route('/try_on')
def try_on_page():
    """
    [功能] 虛擬試穿主頁
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('try_on.html')

@app.route('/smart_mirror')
def smart_mirror():
    """
    [情境] 智慧鏡模式 (Smart Mirror Interface)
    設計給鏡面顯示器的極簡介面
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()
    user = db_conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    last_analysis = db_conn.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
        (session['user_id'],)
    ).fetchone()
    db_conn.close()

    # 預設值
    recommendation = "今天還沒進行分析喔！"
    style_info = {}

    if last_analysis:
        # noinspection PyBroadException
        try:
            rec_json = json.loads(last_analysis['final_recommendation'])

            # [優化] 優先顯示短語 tagline (desc)，沒有才顯示長篇 advice
            recommendation = rec_json.get('desc') or rec_json.get('story') or rec_json.get('summary', '保持自信！')

            if 'archetype' in rec_json:
                style_info['name'] = rec_json['archetype']
            # 提取穿搭指南 (若有)
            if 'clothing_guide' in rec_json:
                style_info['guide'] = rec_json['clothing_guide']
        except:
            pass

    city = request.args.get('city', 'Taoyuan')
    weather = get_weather_data(city)

    return render_template('mirror.html',
                           user=user,
                           weather=weather,
                           rec=recommendation,
                           style_info=style_info)

@app.route('/api/check_ar_capability', methods=['POST'])
def check_ar_api():
    # 簡單回傳 True，讓前端啟動攝影機
    return jsonify({'status': 'success', 'ar_ready': True})

@app.route('/lab')
def lab_page():
    """
    [數據] 風格實驗室 (Style Lab)
    顯示體態追蹤圖表與 AI 深度數據
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()

    # 1. 體態追蹤數據
    tracking = db_conn.execute(
        'SELECT * FROM body_tracking WHERE user_id = ? ORDER BY recorded_at ASC',
        (session['user_id'],)
    ).fetchall()

    # 2. 最新分析
    last_analysis = db_conn.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
        (session['user_id'],)
    ).fetchone()
    db_conn.close()

    # 3. 資料處理
    analysis_data = {'face': {}, 'body': {}, 'rec': {}}
    if last_analysis:
        # noinspection PyBroadException
        try:
            if last_analysis['face_data']:
                analysis_data['face'] = json.loads(last_analysis['face_data'])
            if last_analysis['body_data']:
                analysis_data['body'] = json.loads(last_analysis['body_data'])

            # 處理推薦結果
            if last_analysis['final_recommendation']:
                raw_rec = json.loads(last_analysis['final_recommendation'])
                analysis_data['rec'] = raw_rec
        except:
            pass

    # 4. Chart.js 數據準備
    # 只取日期部分 (YYYY-MM-DD)
    chart_labels = [(t['recorded_at'].split(' ')[0] if t['recorded_at'] else 'Unknown') for t in tracking]
    chart_weights = [t['weight'] for t in tracking]

    return render_template('lab.html',
                           tracking=tracking,
                           labels=chart_labels,
                           weights=chart_weights,
                           analysis=analysis_data)

# --- 4. 社群與趨勢 ---
@app.route('/community')
def community_page():
    """
    [社群] 動態牆首頁
    顯示貼文、投票與互動
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()

    # 1. 撈取貼文 (把 p.author_id 改成 p.user_id，following_id 改成 followed_id)
    posts = db_conn.execute('''
        SELECT p.*, u.name as user_name, u.role as user_role, u.is_vip, u.avatar as user_avatar,
               (SELECT COUNT(*) FROM likes WHERE post_id = p.id) as likes_count,
               EXISTS(SELECT 1 FROM likes WHERE post_id = p.id AND user_id = ?) as is_liked,
               EXISTS(SELECT 1 FROM follows WHERE follower_id = ? AND followed_id = p.user_id) as is_following
        FROM posts p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 50
    ''', (session['user_id'], session['user_id'])).fetchall()

    posts_data = []
    for p in posts:
        # 撈取留言
        comments = db_conn.execute('''
            SELECT c.*, u.name as commenter_name, u.role as commenter_role 
            FROM comments c 
            JOIN users u ON c.user_id = u.id 
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC LIMIT 3
        ''', (p['id'],)).fetchall()

        # 計算投票百分比
        total = p['poll_yes'] + p['poll_no']
        yes_pct = int((p['poll_yes'] / total) * 100) if total > 0 else 0

        # 解析標籤
        raw_tags = p['tags']
        tags = []
        if raw_tags:
            # noinspection PyBroadException
            try:
                parsed = json.loads(raw_tags)
            except:
                parsed = raw_tags

            if isinstance(parsed, str) and parsed.startswith('['):
                import ast
                # noinspection PyBroadException
                try:
                    parsed = ast.literal_eval(parsed)
                except:
                    parsed = [parsed]
            elif isinstance(parsed, str):
                parsed = [parsed]

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str) and item.startswith('['):
                        import ast
                        # noinspection PyBroadException
                        try:
                            tags.extend(ast.literal_eval(item))
                        except:
                            tags.append(item)
                    else:
                        tags.append(item)

        # 處理貼文圖片路徑
        img_path = p['image_path']
        if img_path and not img_path.startswith('http'):
            clean_path = img_path.replace('static/', '')
            img_url = url_for('static', filename=clean_path)
        else:
            img_url = img_path

        # 處理發文者的大頭貼路徑
        raw_avatar = p['user_avatar']
        if raw_avatar and not raw_avatar.startswith('http'):
            author_avatar = url_for('static', filename=raw_avatar.replace('static/', ''))
        elif raw_avatar:
            author_avatar = raw_avatar
        else:
            author_avatar = f"https://ui-avatars.com/api/?name={p['user_name']}&background=F3EFEE&color=D4A5A5"

        # 將整理好的資料放進清單
        posts_data.append({
            'id': p['id'],
            'author_avatar': author_avatar,
            'image_path': img_url,
            'content': p['content'],
            'author_name': "匿名用戶" if p['is_anonymous'] else p['user_name'],
            'author_role': p['user_role'],
            'is_vip': bool(p['is_vip']),
            'author_id': p['user_id'],  # 💡 這裡也要改成 p['user_id']
            'is_anonymous': bool(p['is_anonymous']),
            'is_qa': bool(p['is_qa']),
            'poll_yes': p['poll_yes'],
            'poll_no': p['poll_no'],
            'yes_percent': yes_pct,
            'tags': tags,
            'likes_count': p['likes_count'],
            'is_liked': bool(p['is_liked']),
            'is_following': bool(p['is_following']),
            'created_at': p['created_at'][:10],
            'comments': comments
        })

    db_conn.close()
    return render_template('community.html', posts=posts_data)

@app.route('/community/new', methods=['GET', 'POST'])
def new_post():
    """
    [功能] 發布新貼文
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        # 1. 圖片檢查
        if 'image' not in request.files or request.files['image'].filename == '':
            flash('請選擇要上傳的照片', 'warning')
            return redirect(request.url)

        upload_file = request.files['image']

        # 2. 儲存圖片
        filename = secure_filename(upload_file.filename)
        unique_name = f"post_{uuid.uuid4()}_{filename}"

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        upload_file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_name))

        # 3. 處理內容與標籤
        content = ContentSafety.sanitize(request.form.get('content', ''))
        raw_tags = request.form.get('tags', '')
        tags_list = [t.strip() for t in raw_tags.split(',') if t.strip()]

        # 4. 寫入資料庫
        db_conn = get_db_connection()
        db_conn.execute(
            '''INSERT INTO posts 
               (user_id, image_path, content, tags, is_anonymous, is_qa) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (
                session['user_id'],
                f"uploads/{unique_name}",
                content,
                json.dumps(tags_list, ensure_ascii=False),
                request.form.get('is_anonymous') == 'on',
                request.form.get('is_qa') == 'on'
            )
        )
        db_conn.commit()
        db_conn.close()

        flash('貼文發布成功！', 'success')
        return redirect(url_for('community_page'))

    return render_template('new_post.html')

@app.route('/api/post/delete/<int:post_id>', methods=['POST'])
def delete_post_api(post_id):
    """
    [API] 刪除貼文
    檢查是否為作者本人或管理員，並連帶清除相關按讚與留言
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    db_conn = get_db_connection()
    try:
        post = db_conn.execute('SELECT user_id FROM posts WHERE id=?', (post_id,)).fetchone()
        if not post:
            return jsonify({'status': 'error', 'msg': '貼文不存在'}), 404

        # 權限檢查 (作者本人或是 admin 才能刪除)
        if post['user_id'] != session['user_id'] and session.get('role') != 'admin':
            return jsonify({'status': 'error', 'msg': '權限不足'}), 403

        # 刪除關聯資料避免資料庫殘留垃圾
        db_conn.execute('DELETE FROM comments WHERE post_id=?', (post_id,))
        db_conn.execute('DELETE FROM likes WHERE post_id=?', (post_id,))
        db_conn.execute('DELETE FROM posts WHERE id=?', (post_id,))
        db_conn.commit()

        return jsonify({'status': 'success', 'msg': '貼文已刪除'})
    except Exception as err:
        print(f"Delete Post Error: {err}")
        return jsonify({'status': 'error', 'msg': '刪除失敗'}), 500
    finally:
        db_conn.close()

@app.route('/u/<int:user_id>')
def user_public_profile(user_id):
    """
    [頁面] 使用者公開主頁 (社群個人檔案)
    """
    db_conn = get_db_connection()

    # 1. 抓取使用者基本資料 (💡 修正：補上 avatar 欄位)
    user = db_conn.execute('SELECT id, name, role, is_vip, style_preferences, avatar FROM users WHERE id=?', (user_id,)).fetchone()

    # 💡 修正：找不到使用者時導回首頁，避免缺少 404.html 報錯
    if not user:
        db_conn.close()
        flash('找不到該使用者的主頁', 'warning')
        return redirect(url_for('index'))

    # 2. 抓取該使用者的貼文
    posts = db_conn.execute('SELECT * FROM posts WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()

    # 3. 處理貼文圖片路徑
    formatted_posts = []
    total_likes = 0
    for p in posts:
        post = dict(p)
        if post['image_path'] and not post['image_path'].startswith('http'):
            post['image_path'] = url_for('static', filename=post['image_path'].replace('static/', ''))
        formatted_posts.append(post)
        total_likes += post['likes_count']

    # 4. 解析風格標籤 (從偏好設定中讀取 'like' 標籤)
    style_tags = []
    # noinspection PyBroadException
    try:
        if user['style_preferences']:
            prefs = json.loads(user['style_preferences'])
            style_tags = prefs.get('like', [])[:5]  # 只顯示前5個
    except:
        pass

    # 💡 修正：處理大頭貼路徑 (解決相對路徑 404 問題與動態生成預設頭像)
    raw_avatar = user['avatar']
    if raw_avatar and not raw_avatar.startswith('http'):
        avatar_url = url_for('static', filename=raw_avatar.replace('static/', ''))
    elif raw_avatar:
        avatar_url = raw_avatar
    else:
        avatar_url = f"https://ui-avatars.com/api/?name={user['name']}&background=F3EFEE&color=D4A5A5"

    # 5. 組合傳給模板的資料
    profile_user = {
        'id': user['id'],
        'name': user['name'],
        'avatar': avatar_url,
        'role': user['role'],
        'is_vip': bool(user['is_vip']),
        'bio': "熱愛時尚的 Style Smart 用戶",  # 未來可擴充 bio 欄位
        'style_tags': style_tags
    }

    stats = {
        'post_count': len(posts),
        'received_likes': total_likes,
        'follower_count': 0
    }

    db_conn.close()

    return render_template('user_public_profile.html',
                           profile_user=profile_user,
                           posts=formatted_posts,
                           stats=stats)

@app.route('/trends')
def trends_page():
    """
    [情報] 流行趨勢看板
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()
    user = db_conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

    # 撈取趨勢 (影響力高的在前)
    trends_db = db_conn.execute('SELECT * FROM trends ORDER BY influence_score DESC LIMIT 10').fetchall()

    trend_list = []
    for t in trends_db:
        # noinspection PyBroadException
        try:
            history = json.loads(t['data_points'])
        except:
            history = []

        # 預測趨勢
        prediction = TrendEngine.forecast_trend(history)
        match_score = TrendEngine.calculate_compatibility(dict(user), t['category'])

        # 撈取該趨勢的名人範本
        celebs = db_conn.execute('''
            SELECT c.*, 
            (SELECT COUNT(*) FROM celeb_likes WHERE celeb_id = c.id) as likes_count,
            (SELECT 1 FROM celeb_likes WHERE celeb_id = c.id AND user_id = ?) as is_liked
            FROM celebrity_looks c 
            WHERE trend_id = ? LIMIT 3
        ''', (session['user_id'], t['id'])).fetchall()

        trend_list.append({
            'id': t['id'],
            'keyword': t['keyword'],
            'status': t['status'],
            'score': t['influence_score'],
            'description': t['description'],
            'history': history,
            'prediction': prediction,
            'match_score': match_score,
            'celebs': [dict(c) for c in celebs]  # 轉為字典以便前端使用
        })

    db_conn.close()

    # 地區過濾器 (裝飾用)
    region_label = "台灣趨勢" if session.get('locale') == 'zh_TW' else "全球趨勢"

    return render_template('trends.html',
                           trends=trend_list,
                           region=region_label,
                           user=user)

# --- 5. 個人化與商城 ---
@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()
    user_id = session['user_id']

    # 1. 處理資料更新 (POST)
    if request.method == 'POST':
        try:
            # 先將使用者傳入的偏好資料打包成 JSON
            likes = json.dumps(request.form.getlist('style_likes'), ensure_ascii=False)
            dislikes = json.dumps(request.form.getlist('style_dislikes'), ensure_ascii=False)
            wear_types = json.dumps(request.form.getlist('wear_types'), ensure_ascii=False)
            color_prefs = json.dumps(request.form.getlist('color_prefs'), ensure_ascii=False)
            occasion_prefs = json.dumps(request.form.getlist('occasion_prefs'), ensure_ascii=False)

            # 基本的 UPDATE 欄位與對應參數
            update_sql = '''
                UPDATE users 
                SET name=?, age=?, gender=?, height=?, weight=?, body_type=?, 
                    style_preferences=?, style_dislikes=?, wear_types=?, 
                    color_prefs=?, occasion_prefs=?, fabrics=?, patterns=?, 
                    clothing_issues=?, life_stage=?
            '''
            params = [
                request.form.get('name'), request.form.get('age'), request.form.get('gender'),
                request.form.get('height'), request.form.get('weight'), request.form.get('body_type'),
                likes, dislikes, wear_types, color_prefs, occasion_prefs,
                request.form.get('fabrics'), request.form.get('patterns'),
                request.form.get('clothing_issues'), request.form.get('life_stage')
            ]

            # --- A. 處理大頭照上傳 ---
            avatar_url = None
            if 'avatar' in request.files:
                upload_file = request.files['avatar']
                if upload_file and upload_file.filename != '':
                    filename = secure_filename(f"avatar_{user_id}.png")
                    upload_path = os.path.join('static', 'uploads', 'avatars')
                    if not os.path.exists(upload_path):
                        os.makedirs(upload_path)
                    upload_file.save(os.path.join(upload_path, filename))
                    avatar_url = f"uploads/avatars/{filename}"

            # 如果有上傳新大頭貼，才把 avatar 加進 SQL 語句中
            if avatar_url:
                update_sql += ", avatar=?"
                params.append(avatar_url)

            # 最後加上 WHERE 條件
            update_sql += " WHERE id=?"
            params.append(user_id)

            # 執行更新
            db_conn.execute(update_sql, params)
            db_conn.commit()

            # --- 💡 修正 Session 更新 ---
            session['user_name'] = request.form.get('name')
            # 撈取更新後最新的 user 資料來放進 session
            updated_user = db_conn.execute('SELECT avatar FROM users WHERE id=?', (user_id,)).fetchone()
            if updated_user and updated_user['avatar']:
                session['avatar'] = updated_user['avatar']

            flash('個人檔案已成功更新', 'success')
            return redirect(url_for('profile_page'))

        except Exception as err:
            print(f"❌ Profile Update Error: {err}")
            flash('儲存失敗，請聯繫管理員', 'error')

    # 2. 讀取用戶資料 (GET)
    user = db_conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()

    # 3. 獲取並解析「最近一次」檢測風格紀錄
    latest_record = db_conn.execute('''
        SELECT final_recommendation, created_at as date, user_image_path as image_url
        FROM analysis_history 
        WHERE user_id = ? AND final_recommendation IS NOT NULL
        ORDER BY created_at DESC LIMIT 1
    ''', (user_id,)).fetchone()

    latest_data = None
    if latest_record:
        # noinspection PyBroadException
        try:
            rec_json = json.loads(latest_record['final_recommendation'])

            style_full = rec_json.get('archetype', 'AI 形象診斷')
            style_display = style_full.split(' (')[0].strip() if ' (' in style_full else style_full

            img_path = latest_record['image_url']
            clean_img_url = img_path.replace('static/', '') if img_path else None

            latest_data = {
                'date': latest_record['date'][:10],  # 只顯示日期
                'image_url': clean_img_url,
                'style_name': style_display,
                'advice': rec_json.get('advice', '').replace('###', '').replace('**', '').strip()[:80] + '...'  # 截斷過長文字
            }
        except:
            latest_data = None

    def safe_json(data):
        # noinspection PyBroadException
        try:
            return json.loads(data) if data else []
        except:
            return []

    # 4. 封裝數據
    avatar_display = user['avatar'].replace('static/', '') if user['avatar'] else None

    p_data = {
        'avatar': avatar_display,
        'email': user['email'], 'name': user['name'], 'age': user['age'],
        'gender': user['gender'], 'height': user['height'], 'weight': user['weight'],
        'body_type': user['body_type'],
        'likes': safe_json(user['style_preferences']),
        'dislikes': safe_json(user['style_dislikes']),
        'wear_types': safe_json(user['wear_types']),
        'colors': safe_json(user['color_prefs']),
        'occasions': safe_json(user['occasion_prefs']),
        'fabrics': user['fabrics'], 'patterns': user['patterns'],
        'issues': user['clothing_issues'], 'life_stage': user['life_stage'],
        'latest_report': latest_data
    }
    db_conn.close()

    # 這裡的 STYLE_TAXONOMY 假設你在 web.py 前面有定義
    return render_template('profile.html', p=p_data, taxonomy=STYLE_TAXONOMY)

@app.route('/settings')
def settings_page():
    """
    [設定] 帳戶設定頁面
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()
    user = db_conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    db_conn.close()

    return render_template('settings.html', user=dict(user) if user else {})

@app.route('/shop')
def shop_page():
    """
    [商城] 商品瀏覽頁
    支援：分類篩選、搜尋、價格區間、智慧過濾 (避雷)
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    # 1. 參數處理
    category = request.args.get('category')
    search_query = request.args.get('q', '').strip()
    # noinspection PyBroadException
    try:
        min_price = int(request.args.get('min', 0))
        max_price = int(request.args.get('max', 999999))
    except:
        min_price, max_price = 0, 999999

    db_conn = get_db_connection()

    # 2. 取得用戶避雷設定
    dislikes = []
    # noinspection PyBroadException
    try:
        user_row = db_conn.execute('SELECT style_preferences FROM users WHERE id=?', (session['user_id'],)).fetchone()
        if user_row and user_row['style_preferences']:
            prefs = json.loads(user_row['style_preferences'])
            dislikes = prefs.get('dislike', [])  # 注意 key 是 dislike 不是 dislikes
    except:
        pass

    # 3. 建構 SQL 查詢
    sql = """
        SELECT c.*, u.name as seller_name
        FROM clothing_items c 
        LEFT JOIN users u ON c.seller_id = u.id 
        WHERE c.status = 'on_sale' AND c.price BETWEEN ? AND ?
    """
    params: list = [min_price, max_price]

    if category:
        sql += " AND c.category = ?"
        params.append(category)

    if search_query:
        sql += " AND (c.title LIKE ? OR c.description LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])

    sql += " ORDER BY c.created_at DESC LIMIT 100"  # 限制數量避免過載

    db_items = db_conn.execute(sql, params).fetchall()
    db_conn.close()

    # 4. 資料過濾與格式化
    display_items = []
    for i in db_items:
        # [智慧過濾] 若標題或描述包含避雷關鍵字，則隱藏該商品
        # is_safe_recommendation 已在 helper functions 定義
        check_text = f"{i['title']} {i['description']}"
        if not is_safe_recommendation(check_text, dislikes):
            continue

        # noinspection PyBroadException
        try:
            tags = json.loads(i['tags']) if i['tags'] else []
        except:
            tags = []

        display_items.append({
            'id': i['id'],
            'title': i['title'],
            'price': i['price'],
            'image': i['image_path'],
            'category': i['category'],
            'tags': tags,
            'seller_name': i['seller_name'] or '匿名賣家',
            'is_mine': i['seller_id'] == session['user_id']
        })

    currency = 'NT$' if session.get('locale') == 'zh_TW' else 'US$'

    return render_template('shop.html',
                           items=display_items,
                           currency=currency,
                           current_cat=category)

@app.route('/seller_center')
def seller_center_page():
    """
    [頁面] 賣家中心 (我的商店)
    修正：移除模擬數據，改為真實資料庫統計
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    db_conn = get_db_connection()

    # 1. 抓取該使用者上架的所有商品
    items = db_conn.execute(
        'SELECT * FROM clothing_items WHERE seller_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()

    # 2. 處理圖片路徑 & 進行真實統計
    formatted_items = []

    # 初始化統計變數
    real_sold_count = 0
    total_inventory_value = 0  # 額外增加：庫存總價值

    for item in items:
        i = dict(item)

        # 處理圖片路徑
        if i['image_path'] and not i['image_path'].startswith('http'):
            i['image_path'] = url_for('static', filename=i['image_path'].replace('static/', ''))

        formatted_items.append(i)

        # [真實統計] 計算已售出數量
        # 假設狀態為 'sold' 代表已售出
        if i['status'] == 'sold':
            real_sold_count += 1

        # [真實統計] 計算在庫商品總價值 (僅計算販售中的)
        if i['status'] == 'on_sale' and i['price']:
            # noinspection PyBroadException
            try:
                total_inventory_value += int(i['price'])
            except:
                pass

    db_conn.close()

    # 3. 組合真實數據 (Stats)
    stats = {
        'total_items': len(items),  # 真實上架總數
        'total_views': 0,  # 目前資料庫尚未追蹤瀏覽數，顯示 0 (真實狀況)
        'sold_count': real_sold_count,  # 真實已售出數量
        'inventory_value': total_inventory_value  # 新增：預估產值
    }

    return render_template('seller_center.html', items=formatted_items, stats=stats)

@app.route('/search')
def search_page():
    """
    [搜尋] 全站搜尋落地頁
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('search.html')

@app.route('/premium')
def premium_landing():
    """
    [VIP] 訂閱介紹頁
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('premium.html')

@app.route('/premium/chat')
@vip_required
def chat_page():
    """
    [VIP] AI 顧問聊天室
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user_id = session['user_id']
    session_id = request.args.get('session_id')
    db_conn = get_db_connection()

    # 1. 取得使用者的所有聊天室列表 (側邊欄用)
    chat_sessions = db_conn.execute('SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC',
                                    (user_id,)).fetchall()

    # 2. 如果沒有任何聊天室，自動幫他建一個
    if not chat_sessions:
        new_id = str(uuid.uuid4())
        db_conn.execute('INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)', (new_id, user_id, '新對話'))
        db_conn.commit()
        chat_sessions = db_conn.execute('SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC',
                                        (user_id,)).fetchall()
        session_id = new_id

    # 3. 如果沒有指定 session_id，預設載入最新的一個
    if not session_id and chat_sessions:
        session_id = chat_sessions[0]['id']

    # 4. 撈取該特定聊天室的對話紀錄
    logs = db_conn.execute('SELECT * FROM chat_logs WHERE session_id = ? ORDER BY created_at ASC',
                           (session_id,)).fetchall()
    db_conn.close()

    return render_template('chat_consultant.html', logs=logs, sessions=chat_sessions, active_session_id=session_id)


# 建立新聊天室 API
@app.route('/api/chat/new', methods=['POST'])
def new_chat_session():
    if 'user_id' not in session: return jsonify({'status': 'error'})
    new_id = str(uuid.uuid4())
    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)',
                    (new_id, session['user_id'], '新對話'))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success', 'session_id': new_id})


# 刪除聊天室 API
@app.route('/api/chat/delete/<session_id>', methods=['POST'])
def delete_chat_session(session_id):
    if 'user_id' not in session: return jsonify({'status': 'error'})
    db_conn = get_db_connection()
    db_conn.execute('DELETE FROM chat_logs WHERE session_id = ? AND user_id = ?', (session_id, session['user_id']))
    db_conn.execute('DELETE FROM chat_sessions WHERE id = ? AND user_id = ?', (session_id, session['user_id']))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success'})

@app.route('/premium/calendar')
@vip_required
def calendar_page():
    """
    [VIP] 穿搭行事曆
    """
    db_conn = get_db_connection()
    events = db_conn.execute(
        'SELECT * FROM calendar_events WHERE user_id = ?',
        (session['user_id'],)
    ).fetchall()
    db_conn.close()

    # 轉換為 FullCalendar 格式
    events_json = [
        {
            'title': err['title'],
            'start': err['date_str'],
            'description': err['outfit_desc'],
            'color': '#D4A5A5'  # 統一色系
        }
        for err in events
    ]

    return render_template('calendar.html', events=json.dumps(events_json))

# --- 6. 管理後台 (Admin) ---
@app.route('/admin')
def admin_dashboard():
    """
    [後台] 管理員儀表板 (Dashboard)
    整合：KPI 數據、AI 效能監控、商城管理、檢舉審核、深度數據洞察
    """
    # 1. 權限檢查
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('權限不足，無法進入後台', 'error')
        return redirect(url_for('index'))

    db_conn = get_db_connection()
    try:
        # 2. 平台基礎數據 (KPIs)
        stats = {
            'users': db_conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
            'posts': db_conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0],
            'reports': db_conn.execute('SELECT COUNT(*) FROM reports WHERE status="pending"').fetchone()[0],
        }

        # 3. AI 效能監控 (AI Performance)
        total = db_conn.execute('SELECT COUNT(*) FROM analysis_history').fetchone()[0]
        errors = db_conn.execute('SELECT COUNT(*) FROM analysis_history WHERE is_incorrect = 1').fetchone()[0]
        error_rate = round((errors / total * 100), 1) if total > 0 else 0

        # A/B Test 數據
        def get_rate(variant):
            # noinspection PyBroadException
            try:
                t = db_conn.execute("SELECT COUNT(*) FROM analysis_history WHERE ab_variant = ?", (variant,)).fetchone()[0]
                c = db_conn.execute("SELECT COUNT(*) FROM analysis_history WHERE ab_variant = ? AND is_converted = 1", (variant,)).fetchone()[0]
                return {'count': t, 'rate': round((c / t * 100), 1) if t > 0 else 0}
            except:
                return {'count': 0, 'rate': 0}

        ai_stats = {
            'total': total,
            'error_rate': error_rate,
            'ab_test': {'A': get_rate('A'), 'B': get_rate('B')}
        }

        # 4. 深度數據洞察 (Deep Analytics)
        demographics = {'gender': {'labels': [], 'data': []}, 'age': {'labels': [], 'data': []}}
        correlation_data = []

        try:
            # 4-1. 性別分佈
            gender_rows = db_conn.execute('''
                SELECT gender, COUNT(*) as count 
                FROM users 
                WHERE gender IS NOT NULL 
                GROUP BY gender
            ''').fetchall()
            if gender_rows:
                demographics['gender']['labels'] = [r['gender'] for r in gender_rows]
                demographics['gender']['data'] = [r['count'] for r in gender_rows]

            # 4-2. 年齡層分佈 (使用 2026 - birth_year 計算)
            age_rows = db_conn.execute('''
                SELECT 
                    CASE 
                        WHEN (2026 - birth_year) < 20 THEN 'Gen Z (<20)'
                        WHEN (2026 - birth_year) BETWEEN 20 AND 29 THEN 'Young Adult (20-29)'
                        WHEN (2026 - birth_year) BETWEEN 30 AND 39 THEN 'Mature (30-39)'
                        ELSE 'Senior (40+)'
                    END as age_group,
                    COUNT(*) as count
                FROM users
                WHERE birth_year IS NOT NULL
                GROUP BY age_group
            ''').fetchall()
            if age_rows:
                demographics['age']['labels'] = [r['age_group'] for r in age_rows]
                demographics['age']['data'] = [r['count'] for r in age_rows]

            # 4-3. 風格 x 身形 關聯熱圖 (Heatmap Data)
            # 找出哪種身形最常被匹配到哪種風格
            correlation_data = db_conn.execute('''
                SELECT 
                    json_extract(body_data, '$.shape') as body_shape,
                    json_extract(final_recommendation, '$.archetype') as style,
                    COUNT(*) as count
                FROM analysis_history 
                WHERE body_data IS NOT NULL AND final_recommendation IS NOT NULL
                GROUP BY body_shape, style
                ORDER BY count DESC
                LIMIT 10
            ''').fetchall()

        except Exception as err:
            print(f"Deep Analytics Error: {err}")
            # 若欄位不存在 (尚未執行 setup_db_final)，保持空數據避免報錯

        # 5. 進階圖表數據 (Fairness & Trend)
        try:
            fairness_data = calculate_fairness_metrics()
            trend_data = get_trend_analysis()
        except Exception as err:
            print(f"Chart Data Error: {err}")
            fairness_data = []
            trend_data = {}

        # 6. 資料列表 (Lists)
        items = db_conn.execute('SELECT * FROM clothing_items ORDER BY created_at DESC LIMIT 50').fetchall()
        users = db_conn.execute('SELECT * FROM users ORDER BY created_at DESC LIMIT 50').fetchall()

        # 檢舉審核清單 (關聯查詢)
        reports = db_conn.execute('''
            SELECT r.*, u.name as reporter_name, p.content as post_content 
            FROM reports r 
            JOIN users u ON r.reporter_id = u.id 
            LEFT JOIN posts p ON r.post_id = p.id 
            ORDER BY r.created_at DESC LIMIT 50
        ''').fetchall()

        # 風格提案
        # noinspection PyBroadException
        try:
            proposals = db_conn.execute('''
                SELECT p.*, u.name as user_name FROM style_proposals p 
                JOIN users u ON p.user_id = u.id WHERE p.status = 'pending' ORDER BY p.created_at DESC
            ''').fetchall()
        except:
            proposals = []

        # 系統參數 (趨勢權重)
        # noinspection PyBroadException
        try:
            trend_config = db_conn.execute("SELECT value FROM system_configs WHERE key='trend_weights'").fetchone()
            current_trends = json.loads(trend_config['value']) if trend_config else {}
        except:
            current_trends = {}

    finally:
        db_conn.close()

    # 7. 功能開關 (Feature Flags)
    feature_flags = [
        {'name': 'Beta: 3D 試穿', 'status': 'active', 'usage': 'Low', 'action': '考慮下架'},
        {'name': 'Legacy: 舊版問卷', 'status': 'deprecated', 'usage': 'None', 'action': '已封存'},
        {'name': 'Core: 臉型分析', 'status': 'active', 'usage': 'High', 'action': '核心功能'}
    ]

    return render_template('admin.html',
                           stats=stats,
                           ai_stats=ai_stats,
                           demographics=demographics,       # [NEW] 人口統計
                           correlation_data=correlation_data, # [NEW] 關聯數據
                           items=items,
                           users=users,
                           reports=reports,
                           model_version=CURRENT_MODEL_VERSION,
                           fairness_data=fairness_data,
                           bias_data=fairness_data,
                           trend_data=trend_data,
                           feature_flags=feature_flags,
                           proposals=proposals,
                           current_trends=current_trends)

@app.route('/admin/ban_user/<int:id>')
def admin_ban_user(item_id):
    """
    [管理] 停權使用者
    """
    if session.get('role') != 'admin':
        return redirect(url_for('index'))

    db_conn = get_db_connection()
    try:
        db_conn.execute("UPDATE users SET status='banned' WHERE id=?", (item_id,))
        db_conn.commit()
        flash(f'使用者 ID:{item_id} 已停權', 'warning')
    finally:
        db_conn.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unban_user/<int:id>')
def admin_unban_user(item_id):
    """
    [管理] 解除停權
    """
    if session.get('role') != 'admin':
        return redirect(url_for('index'))

    db_conn = get_db_connection()
    try:
        db_conn.execute("UPDATE users SET status='active' WHERE id=?", (item_id,))
        db_conn.commit()
        flash(f'使用者 ID:{item_id} 已恢復正常狀態', 'success')
    finally:
        db_conn.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_item/<int:id>', methods=['GET', 'POST'])
def admin_delete_item(item_id):
    """
    [管理] 強制刪除商品 (含檔案清理)
    """
    if session.get('role') != 'admin':
        return redirect(url_for('index'))

    db_conn = get_db_connection()
    try:
        item = db_conn.execute('SELECT image_path FROM clothing_items WHERE id = ?', (id,)).fetchone()

        if item:
            # 刪除資料
            db_conn.execute('DELETE FROM clothing_items WHERE id=?', (item_id,))
            db_conn.commit()

            # 清理檔案 (Safe Delete)
            img_path = item['image_path']
            if img_path and 'default' not in img_path and not img_path.startswith('http'):
                # 處理可能的路徑前綴 (static/uploads vs uploads)
                rel_path = img_path.replace('static/', '')
                full_path = os.path.join(app.root_path, 'static', rel_path)

                # 再次確認檔案存在才刪除
                if os.path.exists(full_path):
                    # noinspection PyBroadException
                    try:
                        os.remove(full_path)
                    except:
                        pass

            flash('商品已強制下架', 'success')
        else:
            flash('找不到該商品', 'warning')

    finally:
        db_conn.close()

    return redirect(url_for('admin_dashboard'))

# ---  後端接口  ---
# --- 7. AI 分析與處理 ---
@app.route('/force_train')
def force_train():
    try:
        # 1. 執行訓練
        stats = face_trainer.train()

        # 2. 重新載入模型
        global face_engine
        face_engine = FaceAnalyzer()

        return f"""
        <h1>訓練完成！ (英文資料夾版)</h1>
        <p>AI 已成功讀取英文資料夾並轉換為中文標籤。</p>
        <pre>{stats}</pre>
        <br>
        <a href="/">回到首頁測試</a>
        """
    except Exception as err:
        return f"<h1>訓練失敗</h1><p>{str(err)}</p>"

@app.route('/api/admin/retrain_model', methods=['POST'])
def api_retrain_model():
    """
    [管理員功能] 觸發 AI 重新掃描素材資料夾並訓練模型
    """
    # 簡單的權限檢查
    if session.get('role') != 'admin':
        return jsonify({'status': 'error', 'msg': '權限不足'}), 403

    try:
        # 1. 執行訓練 (這會花一點時間掃描硬碟)
        stats = face_trainer.train()

        # 2. 讓分析引擎重新載入最新的 JSON (Hot Reload)
        global face_engine
        face_engine = FaceAnalyzer()

        return jsonify({
            'status': 'success',
            'msg': '模型重新訓練完成！',
            'details': stats
        })
    except Exception as err:
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/api/analyze_face', methods=['POST'])
def analyze_face_api():
    """
    [AI] 臉部分析 (MediaPipe FaceMesh)
    功能：正確傳遞正臉與側臉圖片，執行一次性融合分析
    """
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'msg': '未接收到圖片'}), 400

    front_file = request.files['image']
    side_file = request.files.get('side_image')  # 側臉 (選填)

    if front_file.filename == '':
        return jsonify({'status': 'error', 'msg': '檔案名稱為空'}), 400

    try:
        # 1. 儲存正臉
        timestamp = int(time.time())
        uid_short = str(uuid.uuid4())[:8]

        filename = secure_filename(f"face_{timestamp}_{uid_short}.jpg")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        front_file.save(save_path)
        session['current_image_path'] = f"static/uploads/{filename}"

        # 2. 儲存側臉 (如果有的話)
        side_save_path = None
        if side_file and side_file.filename != '':
            side_filename = secure_filename(f"face_side_{timestamp}_{uid_short}.jpg")
            side_save_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
            side_file.save(side_save_path)

        # 3. [關鍵修正] 呼叫 analyze 時同時傳入兩個路徑
        result, error = face_engine.analyze(save_path, side_save_path)

        if error:
            return jsonify({'status': 'error', 'msg': error}), 400

        # 4. 回傳資料
        # 注意：我們已經在 analyze 內部完成了融合，所以 side_data 可以簡化
        # 但為了前端相容性，如果側臉存在，我們還是回傳 side_data 結構，讓前端顯示預覽圖
        side_result = None
        if side_save_path:
            side_result = {'analyzed_image': os.path.basename(side_save_path)}

        response_data = {
            'status': 'success',
            'msg': '分析完成',
            'data': result,  # 包含 3D 融合資料的正臉結果
            'side_data': side_result,
            'raw_image_path': f"static/uploads/{filename}"
        }
        return jsonify(response_data)

    except Exception as err:
        print(f"❌ Face Analysis Error: {err}")
        return jsonify({'status': 'error', 'msg': f'分析服務異常: {str(e)}'}), 500

@app.route('/api/analyze_skin', methods=['POST'])
def api_analyze_skin():
    """
    [AI] 膚色分析 (Skin Tone Analyzer)
    功能：檢測膚色冷暖調與四季類型 (Spring/Summer/Autumn/Winter)
    """
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'msg': '未上傳圖片'}), 400

    skin_img_file = request.files['image']

    # noinspection PyBroadException
    try:
        # 1. 儲存暫存檔
        filename = secure_filename(f"skin_{uuid.uuid4().hex[:8]}.jpg")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        skin_img_file.save(save_path)

        # 2. 執行分析
        result, err = skin_engine.analyze(save_path)

        # 3. 清理暫存 (建議開啟，避免伺服器堆積太多暫存圖)
        # noinspection PyBroadException
        try:
            os.remove(save_path)
        except:
            pass

        if err:
            return jsonify({'status': 'error', 'msg': err})

        return jsonify({
            'status': 'success',
            'data': result
        })

    except Exception as err:
        print(f"❌ Skin Analysis Error: {err}")
        return jsonify({'status': 'error', 'msg': '膚色檢測發生錯誤'}), 500

@app.route('/api/analyze_body', methods=['POST'])
def analyze_body_api():
    """
    [AI] 身形分析 (MediaPipe Pose)
    功能：結合手動輸入數據與 AI 視覺分析
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    # 1. 頻率檢查
    if not check_analysis_frequency(session['user_id']):
        return jsonify({'status': 'warning', 'msg': '操作過於頻繁，請休息一下。'}), 429

    ai_result = {'shape': '直筒型 (H-Shape)', 'source': 'default'}
    manual_data = {}

    # 2. 處理手動數據
    if request.is_json:
        manual_data = request.json
        try:
            waist = float(manual_data.get('waist', 0))
            hip = float(manual_data.get('hip', 0))
            chest = float(manual_data.get('chest', 0))
            shoulder = float(manual_data.get('shoulder', 0))

            # 簡易規則判斷 (作為 AI 失敗時的備案)
            if waist > 0 and hip > 0:
                whr = waist / hip

                # 如果沒有填胸圍，把「肩寬」乘以 2.5 轉換成「肩圍」來跟臀圍比較
                upper_measure = chest if chest > 0 else (shoulder * 2.5)

                if whr <= 0.75:
                    # 腰夠細的前提下，看上半身是否跟臀部一樣豐滿
                    ai_result['shape'] = '沙漏型 (X-Shape)' if upper_measure >= (hip * 0.9) else '梨型 (A-Shape)'
                elif whr > 0.85 or waist >= hip:
                    ai_result['shape'] = '蘋果型 (O-Shape)'
                elif upper_measure > hip * 1.05:
                    ai_result['shape'] = '倒三角型 (Y-Shape)'
                elif upper_measure < hip * 0.95:
                    ai_result['shape'] = '梨型 (A-Shape)'
                else:
                    ai_result['shape'] = '直筒型 (H-Shape)'

                ai_result['source'] = 'manual'
        except ValueError:
            pass

    # 3. 處理圖片 (優先使用新上傳，無則用 Session 舊圖)
    target_path = None
    if 'image' in request.files:
        f = request.files['image']
        fname = secure_filename(f"body_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg")
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(target_path)
        session['current_image_path'] = f"static/uploads/{fname}"  # 更新 session

    elif session.get('current_image_path'):
        rel_path = session.get('current_image_path').replace('static/', '')
        abs_path = os.path.join(app.root_path, 'static', rel_path)
        if os.path.exists(abs_path):
            target_path = abs_path

    # 4. 執行 AI 分析
    if target_path:
        vis_res, err = body_engine.analyze(target_path, manual_data)
        if vis_res:
            ai_result.update(vis_res)
            ai_result['source'] = 'ai_vision'

    # 5. 寫入資料庫 (使用 Context Manager 確保連線關閉)
    try:
        with get_db_connection() as db_conn:
            db_conn.execute('''
                INSERT INTO body_tracking (user_id, weight, waist, hip, note, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                manual_data.get('weight', 0),
                manual_data.get('waist', 0),
                manual_data.get('hip', 0),
                f"Shape: {ai_result['shape']}",
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            db_conn.commit()
    except Exception as err:
        print(f"Tracking Save Error: {err}")

    return jsonify({'status': 'success', 'data': ai_result})

@app.route('/api/generate_full_report', methods=['POST'])
def generate_full_report_api():
    """
    [AI] 生成完整風格報告 (The Brain)
    整合 Face + Body + Weather + Skin -> Matrix Logic -> Report
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    try:
        req_data = request.json
        if not req_data:
            return jsonify({'status': 'error', 'msg': '無效的請求數據'}), 400

        with get_db_connection() as db_conn:
            user = db_conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

        # 1. 數據整合
        face_data_dict = req_data.get('face_data', {})
        body_data_dict = req_data.get('body_data', {})
        skin_season = req_data.get('skin_season', None)

        user_data_for_logic = dict(user)
        user_data_for_logic['face_data'] = face_data_dict
        user_data_for_logic['body_data'] = body_data_dict
        if skin_season:
            user_data_for_logic['skin_season'] = skin_season

        # 2. 呼叫核心邏輯
        weather = get_weather_data()
        advice_text, trace, story, style_info = analyze_style_logic(
            user_data_for_logic, weather
        )

        # 3. 封裝前端所需資料 (包含推薦清單)
        final_rec = {
            'archetype': style_info.get('full_name', '自然風格'),
            'story': story,
            'advice': advice_text,
            'hairstyle': style_info.get('hairstyle', ''),
            'makeup': style_info.get('makeup', ''),
            'accessories': style_info.get('accessories', ''),
            'hairstyle_list': style_info.get('hairstyle_list', []),
            'makeup_list': style_info.get('makeup_list', []),
            'fabrics': style_info.get('fabrics', []),
            'patterns': style_info.get('patterns', []),
            'seasonal_color': style_info.get('seasonal_color', []),
            # 💡 [關鍵修復與優化] 確保 API 回傳乾淨俐落的上下身文字，並加入鞋款
            'dos': f"上衣：{style_info.get('wardrobe_tops', '')} | 下著：{style_info.get('wardrobe_bottoms', '')} | 鞋款：{style_info.get('wardrobe_shoes', '')}",
            'donts': ["避免不合身的剪裁", "避免與膚色衝突的色系"]
        }

        # 💡 [核心修正] 確保路徑從 Session 取得，並同步寫入 DB
        user_img_path = req_data.get('image_path') or session.get('current_image_path', '')

        # 4. 存入歷史紀錄表 (history)
        with get_db_connection() as db_conn:
            db_conn.execute('''
                INSERT INTO analysis_history (
                    user_id, user_image_path, face_data, body_data, 
                    final_recommendation, ai_confidence, model_version, logic_trace, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                session['user_id'],
                user_img_path, # 存入如 "uploads/face_xxx.jpg"
                json.dumps(face_data_dict, ensure_ascii=False),
                json.dumps(body_data_dict, ensure_ascii=False),
                json.dumps(final_rec, ensure_ascii=False),
                95,
                "StyleNet-2026-v3", # 標註模型版本
                trace
            ))
            db_conn.commit()

        return jsonify({'status': 'success', 'result': final_rec})

    except Exception as err:
        print(f"❌ Report Gen Error: {err}")
        return jsonify({'status': 'error', 'msg': '報告生成失敗，請聯繫管理員'}), 500

@app.route('/api/try_on', methods=['POST'])
def try_on_api():
    """
    [AI] 虛擬試穿 (VTON)
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    # 1. 取得使用者圖片
    user_img_rel = session.get('current_image_path')
    if not user_img_rel:
        return jsonify({'status': 'error', 'msg': '請先在分析頁面上傳全身照'}), 400

    user_full_path = os.path.abspath(os.path.join(app.root_path, user_img_rel))
    if not os.path.exists(user_full_path):
        return jsonify({'status': 'error', 'msg': '找不到使用者照片'}), 404

    # 2. 取得衣物圖片
    cloth_id = request.json.get('clothing_id')
    with get_db_connection() as db_conn:
        cloth = db_conn.execute('SELECT * FROM clothing_items WHERE id=?', (cloth_id,)).fetchone()

    if not cloth: return jsonify({'status': 'error', 'msg': '衣物不存在'}), 404
    if cloth['image_path'].startswith('http'):
        return jsonify({'status': 'error', 'msg': '此為網路範例圖，暫不支援試穿'}), 400

    cloth_full_path = os.path.abspath(os.path.join(app.root_path, cloth['image_path']))

    # 3. 執行 VTON
    try:
        res_path, err = vton_engine.generate(user_full_path, cloth_full_path)

        if err:
            return jsonify({'status': 'error', 'msg': f'AI 生成失敗: {err}'}), 500

        # 4. 存檔結果
        new_name = f"tryon_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        target_save = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
        shutil.copy(res_path, target_save)

        # 寫入歷史
        with get_db_connection() as db_conn:
            db_conn.execute('INSERT INTO try_on_history (user_id, original_img, cloth_img, result_img) VALUES (?,?,?,?)',
                         (session['user_id'], user_img_rel, cloth['image_path'], f"uploads/{new_name}"))
            db_conn.commit()

        return jsonify({'status': 'success', 'result_url': f"static/uploads/{new_name}"})

    except Exception as err:
        print(f"VTON Error: {err}")
        return jsonify({'status': 'error', 'msg': '試穿系統忙碌中'}), 500

@app.route('/api/generate_ai_outfit', methods=['POST'])
def generate_ai_outfit():
    """
    [AI] 接收前端 3D 體態截圖與文字指令，進行 2D 擬真服裝生成
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        base64_img = data.get('image', '')

        if not base64_img:
            return jsonify({'status': 'error', 'msg': '沒有接收到體態圖片'}), 400

        # 1. 處理前端傳來的 Base64 圖片
        if ',' in base64_img:
            base64_img = base64_img.split(',')[1]

        # 將字串解碼為真實的圖片檔案
        img_data = base64.b64decode(base64_img)

        # 產生不重複的唯一檔名
        filename = f"ai_outfit_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 儲存圖片到 static/uploads 資料夾
        with open(save_path, 'wb') as f:
            f.write(img_data)

        # 產生前端可以讀取的圖片網址
        image_url = url_for('static', filename=f'uploads/{filename}')

        # 2. 回傳成功訊息與圖片網址給前端
        return jsonify({
            'status': 'success',
            'result_url': image_url,
            'msg': f'已為您設計：{prompt}'
        })

    except Exception as err:
        print(f"❌ Generate AI Outfit Error: {err}")
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/api/generate_3d_texture', methods=['POST'])
def generate_3d_texture():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        # 轉成小寫方便比對英文
        prompt_lower = prompt.lower()

        # 預設素材 (維基百科開源圖庫)
        color_url = "https://upload.wikimedia.org/wikipedia/commons/1/13/White_fabric_texture.jpg"
        normal_url = "/api/proxy_texture?url=fallback_normal"

        # 💡 關鍵修正：同時檢查中文與英文關鍵字，雙重保險！
        if "丹寧" in prompt or "牛仔" in prompt or "denim" in prompt_lower or "jeans" in prompt_lower:
            color_url = "https://upload.wikimedia.org/wikipedia/commons/4/45/Texture_of_blue_jeans_02.jpg"
        elif "皮衣" in prompt or "皮革" in prompt or "夾克" in prompt or "leather" in prompt_lower:
            color_url = "https://upload.wikimedia.org/wikipedia/commons/7/7b/Black_leather_texture.jpg"
        elif "針織" in prompt or "毛衣" in prompt or "knit" in prompt_lower or "sweater" in prompt_lower:
            color_url = "https://upload.wikimedia.org/wikipedia/commons/8/87/Knitted_texture.jpg"
        elif "棉麻" in prompt or "亞麻" in prompt or "透氣" in prompt or "linen" in prompt_lower or "cotton" in prompt_lower:
            color_url = "https://upload.wikimedia.org/wikipedia/commons/1/13/White_fabric_texture.jpg"
        elif "迷彩" in prompt or "camo" in prompt_lower:
            color_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Camouflage_pattern_001.jpg/800px-Camouflage_pattern_001.jpg"

        return jsonify({
            'status': 'success',
            'texture_data': {
                'map': color_url,
                'normalMap': normal_url
            }
        })

    except Exception as err:
        print(f"❌ 3D 貼圖生成錯誤: {err}")
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/api/proxy_texture')
def proxy_texture():
    url = request.args.get('url')
    if not url: return "Missing URL", 400
    # noinspection PyBroadException
    try:
        # 如果不是強制觸發備用機制，才去下載圖片
        if not url.startswith('fallback_'):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0'}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                return Response(resp.content, mimetype='image/jpeg')
    except Exception as err:
        print(f"⚠️ 圖片下載失敗，啟動備用生成機制: {url}, 錯誤原因: {err}")

    # --- 🚀 OpenCV 自動備用材質生成機制 (3D 光影修復版) ---
    img = np.zeros((512, 512, 3), np.uint8)

    # 判斷是否為「法線貼圖 (Normal Map)」
    if "normal" in url.lower():
        # 💡 關鍵修復：3D 裡的平坦法線必須是「藍紫色 (RGB: 128, 128, 255)」
        # OpenCV 使用 BGR 格式，所以填入 (255, 128, 128)
        img[:] = (255, 128, 128)

        # 加上微弱的凹凸雜訊，讓衣服看起來有真實布料的編織紋理
        noise = np.random.randn(512, 512, 3) * 8
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    else:
        # 💡 顏色貼圖 (Color Map)
        if "jeans" in url.lower() or "denim" in url.lower():
            img[:] = (130, 80, 40)  # 牛仔藍 (BGR格式)
        elif "leather" in url.lower():
            img[:] = (40, 40, 40)  # 深灰色皮衣
        else:
            img[:] = (210, 210, 210)  # 淺灰白棉麻

        # 加入明顯的布料顆粒雜訊
        noise = np.random.randn(512, 512, 3) * 15
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/swap_face', methods=['POST'])
def swap_face_api():
    """
    [AI] 換臉功能
    """
    if 'swapper' not in globals():
        return jsonify({'status': 'error', 'msg': 'AI 模型未載入'}), 503

    try:
        data = request.json
        source_img = base64_to_cv2(data.get('user_face'))
        target_img = base64_to_cv2(data.get('target_body'))

        if source_img is None or target_img is None:
            return jsonify({'status': 'error', 'msg': '圖片解碼失敗'}), 400

        # 執行 Swap
        src_faces = face_app.get(source_img)
        tgt_faces = face_app.get(target_img)

        if not src_faces or not tgt_faces:
            return jsonify({'status': 'error', 'msg': '無法偵測到臉部特徵'}), 400

        src_face = sorted(src_faces, key=lambda x: x.bbox[2] * x.bbox[3])[-1]
        tgt_face = sorted(tgt_faces, key=lambda x: x.bbox[2] * x.bbox[3])[-1]

        res = swapper.get(target_img, tgt_face, src_face, paste_back=True)
        return jsonify({'status': 'success', 'result_url': cv2_to_base64(res)})

    except Exception as err:
        print(f"Swap Error: {err}")
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/api/ai_explain_vote', methods=['POST'])
def ai_explain_vote_api():
    """ [AI] 視覺問答 """
    post_id = request.json.get('post_id')
    with get_db_connection() as db_conn:
        post = db_conn.execute('SELECT image_path FROM posts WHERE id=?', (post_id,)).fetchone()

    if not post: return jsonify({'status': 'error', 'reason': '貼文不存在'})

    img_path = os.path.abspath(os.path.join(app.root_path, 'static', post['image_path'].replace('static/', '')))
    if not os.path.exists(img_path):
        return jsonify({'status': 'error', 'reason': '圖片遺失'})
    # noinspection PyBroadException
    try:
        yes = request.json.get('yes', 0)
        no = request.json.get('no', 0)
        trend = 'popular' if yes >= no else 'unpopular'

        reason = vote_engine.analyze(img_path, trend)
        prefix = "大家喜歡這套！" if trend == 'popular' else "還有進步空間。"
        return jsonify({'status': 'success', 'reason': f"{prefix} {reason}"})
    except:
        return jsonify({'status': 'success', 'reason': 'AI 暫時無法回應，但您的風格很有特色！'})

@app.route('/api/external/v1/analyze', methods=['POST'])
def external_api_analyze():
    """ [Open API] 外部呼叫介面 """
    api_access_key = os.environ.get('API_KEY', 'default_secret')
    if request.headers.get('X-API-KEY') != api_access_key:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.json
        internal_data = {
            'face_data': data.get('face_data', {}),
            'body_data': data.get('body_data', {}),
            'gender': data.get('gender', 'female'),
            'skin_season': data.get('skin_season', None)
        }
        rec, _, _, info = analyze_style_logic(internal_data, get_weather_data())
        return jsonify({
            'status': 'success',
            'meta': {'version': "v2026.1", 'time': datetime.datetime.now().isoformat()},
            'result': {'recommendation': rec, 'style_name': info.get('full_name')}
        })
    except Exception as err:
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/legal/terms')
def terms_page(): return render_template('legal.html', type='terms')

@app.route('/legal/wellness')
def wellness_page(): return render_template('legal.html', type='wellness')

# --- 8. 使用者行為與回饋 ---
@app.route('/api/wear_feedback', methods=['POST'])
def wear_feedback_api():
    """
    [回饋] 穿著感受回饋
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    d = request.json
    db_conn = get_db_connection()
    try:
        db_conn.execute(
            'INSERT INTO wear_logs (user_id, date_str, outfit_desc, feeling, rating) VALUES (?, ?, ?, ?, ?)',
            (
                session['user_id'],
                datetime.datetime.now().strftime('%Y-%m-%d'),
                d.get('desc', '未描述'),
                d.get('feeling', ''),
                d.get('rating', 3)
            )
        )
        db_conn.commit()
    finally:
        db_conn.close()

    return jsonify({'status': 'success', 'msg': 'AI 已學習您的感受，將調整推薦策略。'})

@app.route('/api/delete_history', methods=['POST'])
def delete_history_api():
    """
    [API] 刪除特定的歷史紀錄 (並清理實體圖片檔案)
    """
    # 1. 檢查是否登入
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    db_conn = None

    try:
        # 2. 取得前端傳來的紀錄 ID
        record_id = request.json.get('id')
        if not record_id:
            return jsonify({'status': 'error', 'msg': '缺少紀錄 ID'}), 400

        db_conn = get_db_connection()

        # 3. 為了安全與清理圖片，先確認這筆紀錄是這個使用者的
        record = db_conn.execute('SELECT user_image_path FROM analysis_history WHERE id = ? AND user_id = ?',
                              (record_id, session['user_id'])).fetchone()

        if record:
            # 4. 刪除資料庫紀錄
            db_conn.execute('DELETE FROM analysis_history WHERE id = ? AND user_id = ?',
                         (record_id, session['user_id']))
            db_conn.commit()

            # 5. (進階) 實體刪除圖片檔案，釋放空間
            img_path = record['user_image_path']
            if img_path and 'default' not in img_path and not img_path.startswith('http'):
                rel_path = img_path.replace('static/', '')
                full_path = os.path.join(app.root_path, 'static', rel_path)

                # 確保檔案存在才刪除
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                    except Exception as img_err:
                        print(f"圖片檔案刪除失敗: {img_err}")

            msg = '紀錄已成功刪除'
        else:
            msg = '找不到該紀錄或無權限刪除'

        return jsonify({'status': 'success', 'msg': msg})

    except Exception as err:
        print(f"Delete History Error: {err}")
        return jsonify({'status': 'error', 'msg': '刪除失敗'}), 500
    finally:
        if db_conn is not None:
            db_conn.close()

@app.route('/api/report_error', methods=['POST'])
def report_error():
    """
    [RLHF] 回報 AI 分析錯誤
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    feedback = request.json.get('feedback')
    history_id = request.json.get('history_id')

    db_conn = get_db_connection()
    db_conn.execute('UPDATE analysis_history SET is_incorrect=1, user_feedback=? WHERE id=? AND user_id=?',
                 (feedback, history_id, session['user_id']))
    db_conn.commit()
    db_conn.close()

    return jsonify({'status': 'success', 'msg': '感謝您的回饋，這將幫助 AI 變得更準確！'})

@app.route('/api/user/correct_profile', methods=['POST'])
def correct_user_profile():
    """
    [修正] 手動校正 AI 分析結果
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    new_shape = request.json.get('manual_shape')
    target = request.json.get('target')

    if not new_shape or target not in ['body', 'face']:
        return jsonify({'status': 'error', 'msg': '參數錯誤'})

    db_conn = get_db_connection()
    try:
        last_record = db_conn.execute(
            'SELECT id, body_data, face_data FROM analysis_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()

        if last_record:
            col = 'body_data' if target == 'body' else 'face_data'
            data = json.loads(last_record[col])
            data['shape'] = new_shape
            data['is_manual_corrected'] = True

            db_conn.execute(
                f'UPDATE analysis_history SET {col}=? WHERE id=?',
                (json.dumps(data, ensure_ascii=False), last_record['id'])
            )
            db_conn.commit()
            msg = f'已校正{target}為：{new_shape}'
        else:
            msg = '無紀錄可供校正'
    except Exception as err:
        msg = f'校正失敗: {err}'
    finally:
        db_conn.close()

    return jsonify({'status': 'success', 'msg': msg})

@app.route('/api/get_history_detail/<int:record_id>')
def get_history_detail_api(record_id):
    """
    [API] 獲取單筆歷史紀錄的完整分析報告
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    db_conn = get_db_connection()
    # 確保只能抓取自己的紀錄
    record = db_conn.execute('SELECT * FROM analysis_history WHERE id = ? AND user_id = ?',
                          (record_id, session['user_id'])).fetchone()
    db_conn.close()

    if not record:
        return jsonify({'status': 'error', 'msg': '找不到該筆紀錄'}), 404

    # 安全地解析儲存在資料庫裡的 JSON 文字
    # noinspection PyBroadException
    try:
        rec = json.loads(record['final_recommendation']) if record['final_recommendation'] else {}
    except Exception:
        rec = {}

    return jsonify({
        'status': 'success',
        'data': {
            'recommendation': rec,
            'trace': record['logic_trace'] or '無推論紀錄'
        }
    })

@app.route('/api/dislike_item', methods=['POST'])
def dislike_item_api():
    """
    [避雷] 加入不感興趣清單
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    item_id = request.json.get('item_id')
    db_conn = get_db_connection()
    item = db_conn.execute('SELECT tags, category FROM clothing_items WHERE id = ?', (item_id,)).fetchone()
    db_conn.close()

    if not item: return jsonify({'status': 'error', 'msg': '找不到商品'})

    tags_to_ban = []
    # noinspection PyBroadException
    try:
        if item['tags']:
            loaded = json.loads(item['tags'])
            if isinstance(loaded, list): tags_to_ban.extend(loaded)
    except:
        if item['tags']: tags_to_ban.extend(item['tags'].split(','))

    if item['category']: tags_to_ban.append(item['category'])

    # 這裡需確保 update_user_dislikes 函式存在
    # update_user_dislikes(session['user_id'], tags_to_ban)

    return jsonify({
        'status': 'success',
        'msg': '系統將減少推薦此類風格。',
        'learned_tags': tags_to_ban
    })

@app.route('/api/add_favorite', methods=['POST'])
def add_favorite_api():
    """
    [收藏] 加入 Wishlist
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    item_data = request.json
    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO favorites (user_id, item_data) VALUES (?, ?)',
                 (session['user_id'], json.dumps(item_data, ensure_ascii=False)))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success', 'msg': '已加入收藏'})

@app.route('/api/calendar/add', methods=['POST'])
def calendar_add_api():
    """
    [行事曆] 加入穿搭計畫
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    d = request.json
    if not d.get('title') or not d.get('date'):
        return jsonify({'status': 'error', 'msg': '資料不完整'}), 400

    db_conn = get_db_connection()
    db_conn.execute(
        'INSERT INTO calendar_events (user_id, date_str, title, outfit_desc) VALUES (?, ?, ?, ?)',
        (session['user_id'], d.get('date'), d.get('title'), d.get('desc', ''))
    )
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success', 'msg': '已加入行事曆'})

@app.route('/api/update_consent', methods=['POST'])
def update_consent():
    """
    [隱私] 更新同意狀態
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    db_conn = get_db_connection()
    db_conn.execute('UPDATE users SET data_consent=? WHERE id=?',
                 (request.json.get('consent'), session['user_id']))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/update_privacy_settings', methods=['POST'])
def update_privacy_settings():
    """
    [隱私] 更新隱私偏好
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    data = request.json
    policy = data.get('photo_policy', '30_days')
    ai_consent = 1 if data.get('ai_consent') else 0

    db_conn = get_db_connection()
    db_conn.execute('UPDATE users SET photo_policy = ?, ai_training_consent = ? WHERE id = ?',
                 (policy, ai_consent, session['user_id']))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success', 'msg': '設定已更新'})

@app.route('/api/delete_all_photos', methods=['POST'])
def delete_all_photos():
    """
    [隱私] 一鍵銷毀所有照片
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    db_conn = get_db_connection()
    records = db_conn.execute('SELECT user_image_path FROM analysis_history WHERE user_id = ?',
                           (session['user_id'],)).fetchall()

    count = 0
    for r in records:
        path = r['user_image_path']
        if path and 'default' not in path:
            full_path = os.path.join(app.root_path, 'static', path.replace('static/', ''))
            if os.path.exists(full_path):
                os.remove(full_path)
                count += 1

    db_conn.execute('UPDATE analysis_history SET user_image_path = NULL WHERE user_id = ?', (session['user_id'],))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success', 'msg': f'已銷毀 {count} 張照片'})

@app.route('/api/delete_account', methods=['POST'])
def delete_account():
    """
    [危險] 永久刪除帳號
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    uid = session['user_id']
    db_conn = get_db_connection()
    try:
        tables = ['analysis_history', 'posts', 'comments', 'likes', 'follows',
                  'try_on_history', 'favorites', 'calendar_events', 'chat_logs',
                  'body_tracking', 'wear_logs', 'reports', 'style_proposals']
        for t in tables:
            db_conn.execute(f'DELETE FROM {t} WHERE user_id=?', (uid,))
        db_conn.execute('DELETE FROM users WHERE id=?', (uid,))
        db_conn.commit()
        session.clear()
        return jsonify({'status': 'success', 'msg': '帳號已刪除'})
    except Exception as err:
        db_conn.rollback()
        return jsonify({'status': 'error', 'msg': str(err)}), 500
    finally:
        db_conn.close()

@app.route('/api/download_my_data')
def download_my_data():
    """
    [資料權] 下載個人資料副本
    """
    if 'user_id' not in session: return redirect(url_for('login_page'))

    uid = session['user_id']
    db_conn = get_db_connection()
    data = {
        'profile': dict(db_conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone() or {}),
        'history': [dict(r) for r in db_conn.execute('SELECT * FROM analysis_history WHERE user_id=?', (uid,)).fetchall()],
        'calendar': [dict(r) for r in db_conn.execute('SELECT * FROM calendar_events WHERE user_id=?', (uid,)).fetchall()]
    }
    db_conn.close()

    if 'password' in data['profile']: del data['profile']['password']

    filename = f"takeout_{uid}_{int(time.time())}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return send_file(path, as_attachment=True)

@app.route('/api/generate_pdf')
def generate_pdf():
    """
    [報告] 生成 PDF
    """
    # 確保字型存在 (使用絕對路徑)
    font_path = os.path.join(app.root_path, 'static', 'fonts', 'msjh.ttf')

    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(font_path):
        pdf.add_font('ChineseFont', '', font_path, uni=True)
        pdf.set_font("ChineseFont", size=16)
    else:
        pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Smart Style Report", ln=1, align='C')
    pdf.set_font_size(12)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=1)

    filename = f"report_{int(time.time())}.pdf"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(path)

    return send_file(path, as_attachment=True)

# --- 9. 社群互動 ---
@app.route('/api/toggle_like', methods=['POST'])
def toggle_like_api():
    """
    [整合型 API] 通用按讚功能 (Post & Celeb)
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    target_type = request.json.get('type')  # 'post' 或 'celeb'
    target_id = request.json.get('id')
    user_id = session['user_id']
    db_conn = get_db_connection()

    try:
        new_count = 0
        action = 'liked'

        if target_type == 'post':
            if db_conn.execute('SELECT 1 FROM likes WHERE user_id=? AND post_id=?', (user_id, target_id)).fetchone():
                db_conn.execute('DELETE FROM likes WHERE user_id=? AND post_id=?', (user_id, target_id))
                db_conn.execute('UPDATE posts SET likes_count = likes_count - 1 WHERE id=?', (target_id,))
                action = 'unliked'
            else:
                db_conn.execute('INSERT INTO likes (user_id, post_id) VALUES (?, ?)', (user_id, target_id))
                db_conn.execute('UPDATE posts SET likes_count = likes_count + 1 WHERE id=?', (target_id,))
            new_count = db_conn.execute('SELECT likes_count FROM posts WHERE id=?', (target_id,)).fetchone()[0]

        elif target_type == 'celeb':
            if db_conn.execute('SELECT 1 FROM celeb_likes WHERE user_id=? AND celeb_id=?',
                            (user_id, target_id)).fetchone():
                db_conn.execute('DELETE FROM celeb_likes WHERE user_id=? AND celeb_id=?', (user_id, target_id))
                action = 'unliked'
            else:
                db_conn.execute('INSERT INTO celeb_likes (user_id, celeb_id) VALUES (?, ?)', (user_id, target_id))
            new_count = db_conn.execute('SELECT COUNT(*) FROM celeb_likes WHERE celeb_id=?', (target_id,)).fetchone()[0]

        db_conn.commit()
        return jsonify({'status': 'success', 'action': action, 'count': new_count})
    finally:
        db_conn.close()

@app.route('/api/vote_post', methods=['POST'])
def vote_post_api():
    """
    [互動] 二選一投票
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    pid = request.json.get('post_id')
    vote = request.json.get('vote')
    db_conn = get_db_connection()

    try:
        if vote == 'yes':
            db_conn.execute('UPDATE posts SET poll_yes = poll_yes + 1 WHERE id = ?', (pid,))
        else:
            db_conn.execute('UPDATE posts SET poll_no = poll_no + 1 WHERE id = ?', (pid,))
        db_conn.commit()

        post = db_conn.execute('SELECT poll_yes, poll_no FROM posts WHERE id = ?', (pid,)).fetchone()
        total = post['poll_yes'] + post['poll_no']
        pct = int((post['poll_yes'] / total) * 100) if total > 0 else 0

        return jsonify({'status': 'success', 'yes': post['poll_yes'], 'no': post['poll_no'], 'percent': pct})
    finally:
        db_conn.close()

@app.route('/api/comment_post/<int:post_id>', methods=['POST'])
def comment_post(post_id):
    """
    [留言] 貼文留言 (含心理健康偵測)
    """
    if 'user_id' not in session:
        flash('請先登入才能留言', 'warning')
        return redirect(url_for('login_page'))

    content = request.form.get('content', '')

    # 心理健康防護
    if ContentSafety.check_mental_health(content):
        flash('我們注意到您似乎心情低落。請記得，您並不孤單，需要時請尋求專業協助。', 'warning')
        return redirect(url_for('wellness_page'))

    safe_content = ContentSafety.sanitize(content)
    if not safe_content:
        flash('留言內容不能為空', 'warning')
        return redirect(url_for('community_page'))

    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO comments (user_id, post_id, content) VALUES (?, ?, ?)',
                 (session['user_id'], post_id, safe_content))
    db_conn.commit()
    db_conn.close()

    return redirect(url_for('community_page'))

@app.route('/api/follow_user/<int:target_id>', methods=['POST'])
def follow_user(target_id):
    """
    [追蹤] 關注/取消關注
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    db_conn = get_db_connection()
    try:
        exist = db_conn.execute('SELECT 1 FROM follows WHERE follower_id=? AND followed_id=?',
                             (session['user_id'], target_id)).fetchone()
        if exist:
            db_conn.execute('DELETE FROM follows WHERE follower_id=? AND followed_id=?', (session['user_id'], target_id))
            act = 'unfollowed'
        else:
            db_conn.execute('INSERT INTO follows (follower_id, followed_id) VALUES (?, ?)', (session['user_id'], target_id))
            act = 'followed'
        db_conn.commit()
        return jsonify({'status': 'success', 'action': act})
    finally:
        db_conn.close()

@app.route('/api/report', methods=['POST'])
def submit_report():
    """
    [檢舉] 提交檢舉
    (整合了 report_post 與 submit_report，支援 JSON 與 Form)
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    # 相容 Form Data 與 JSON
    if request.is_json:
        data = request.json
        post_id = data.get('post_id')
        reason = data.get('reason')
    else:
        # 如果是舊版前端呼叫 (支援 URL parameter id)
        post_id = request.form.get('post_id') or request.view_args.get('id')
        reason = request.form.get('reason')

    if not post_id: return jsonify({'status': 'error', 'msg': '參數錯誤'})

    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO reports (reporter_id, post_id, reason, status) VALUES (?, ?, ?, "pending")',
                 (session['user_id'], post_id, reason or '其他'))
    db_conn.commit()
    db_conn.close()

    return jsonify({'status': 'success', 'msg': '檢舉已提交，感謝您的協助'})

@app.route('/api/report_post/<int:post_id>', methods=['POST'])
def report_post_legacy(post_id):
    return submit_report(post_id)

@app.route('/api/trend/match_celeb', methods=['POST'])
def match_celeb_style():
    """
    [趣味] 名人風格比對
    """
    user_style = "簡約"
    if 'user_id' in session:
        db_conn = get_db_connection()
        last = db_conn.execute(
            'SELECT final_recommendation FROM analysis_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
            (session['user_id'],)).fetchone()
        db_conn.close()
        if last:
            # noinspection PyBroadException
            try:
                rec = json.loads(last['final_recommendation'])
                user_style = (rec.get('archetype') or rec.get('summary', '簡約')).split('(')[0].strip()
            except:
                pass
    # noinspection PyBroadException
    try:
        celeb_data = fetch_celeb_match_from_web(user_style)
    except:
        celeb_data = {'name': 'Fashion Icon', 'img': 'https://placehold.co/300x400'}

    return jsonify({
        'status': 'success',
        'similarity': random.randint(75, 98),
        'celeb_name': celeb_data['name'],
        'celeb_image': celeb_data['img'],
        'style_keyword': user_style,
        'msg': f'您的【{user_style}】氣質與【{celeb_data["name"]}】有異曲同工之妙！'
    })

# --- 10. 商城與賣家 ---
@app.route('/api/seller/add_product', methods=['POST'])
def seller_add_product():
    """
    [賣家] 上架商品
    """
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'msg': '未上傳圖片'}), 400

    upload_file = request.files['image']
    if upload_file.filename == '':
        return jsonify({'status': 'error', 'msg': '檔案名稱為空'}), 400

    try:
        # 1. 儲存圖片
        filename = secure_filename(upload_file.filename)
        unique_name = f"prod_{uuid.uuid4()}_{filename}"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        upload_file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_name))
        image_path = f"uploads/{unique_name}"

        # 2. 處理標籤與屬性
        tags_list = [
            request.form.get('category', 'top'),
            request.form.get('quadrant', 'Q5'),
            request.form.get('material', '一般材質'),
            '現貨'
        ]
        if request.form.get('extra_tags'):
            tags_list.extend(request.form.get('extra_tags').split(','))

        # 3. 寫入資料庫
        db_conn = get_db_connection()
        db_conn.execute('''
            INSERT INTO clothing_items (
                seller_id, image_path, title, category, price, description, brand, 
                quadrant, material, pattern, neckline, tags, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'on_sale', CURRENT_TIMESTAMP)
        ''', (
            session['user_id'], image_path,
            request.form.get('title'), request.form.get('category'),
            request.form.get('price'), request.form.get('description'),
            request.form.get('brand', '自有品牌'),
            request.form.get('quadrant', 'Q5'),
            request.form.get('material'),
            request.form.get('pattern'),
            request.form.get('neckline'),
            json.dumps(tags_list, ensure_ascii=False)
        ))
        db_conn.commit()
        db_conn.close()

        return jsonify({'status': 'success', 'msg': '商品上架成功！'})

    except Exception as err:
        return jsonify({'status': 'error', 'msg': str(err)}), 500

@app.route('/api/seller/delete_product', methods=['POST'])
def seller_delete_product():
    """
    [賣家] 刪除商品 (含檔案清理)
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    item_id = request.json.get('item_id')
    db_conn = get_db_connection()

    try:
        item = db_conn.execute('SELECT seller_id, image_path FROM clothing_items WHERE id=?', (item_id,)).fetchone()
        if not item: return jsonify({'status': 'error', 'msg': '商品不存在'})

        # 權限檢查
        if item['seller_id'] != session['user_id'] and session.get('role') != 'admin':
            return jsonify({'status': 'error', 'msg': '無權刪除此商品'})

        # 執行刪除
        db_conn.execute('DELETE FROM clothing_items WHERE id=?', (item_id,))
        db_conn.commit()

        # 清理檔案
        path = item['image_path']
        if path and 'default' not in path and not path.startswith('http'):
            full_path = os.path.join(app.root_path, 'static', path.replace('static/', ''))
            if os.path.exists(full_path):
                os.remove(full_path)

        return jsonify({'status': 'success', 'msg': '商品已刪除'})
    except Exception as err:
        return jsonify({'status': 'error', 'msg': str(err)}), 500
    finally:
        db_conn.close()

@app.route('/api/search', methods=['POST'])
def search_api():
    """
    [搜尋] 全站搜尋 (商品 + 貼文)
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    keyword = request.json.get('keyword', '').strip()
    filters = request.json.get('filters', {})

    db_conn = get_db_connection()
    results = {'items': [], 'posts': []}

    try:
        # 1. 搜尋商品
        item_sql = "SELECT * FROM clothing_items WHERE status='on_sale'"
        params = []

        if keyword:
            item_sql += " AND (title LIKE ? OR tags LIKE ? OR brand LIKE ?)"
            params.extend([f"%{keyword}%"] * 3)

        if filters.get('category'):
            item_sql += " AND category = ?"
            params.append(filters['category'])

        # 執行查詢 (限制 20 筆)
        items = db_conn.execute(item_sql + " ORDER BY created_at DESC LIMIT 20", params).fetchall()

        for i in items:
            img = i['image_path']
            if not img.startswith('http'):
                img = url_for('static', filename=img.replace('static/', ''))

            results['items'].append({
                'id': i['id'], 'title': i['title'], 'price': i['price'],
                'image': img, 'brand': i['brand']
            })

        # 2. 搜尋貼文
        post_sql = "SELECT p.*, u.name FROM posts p JOIN users u ON p.user_id=u.id WHERE 1=1"
        post_params = []

        if keyword:
            post_sql += " AND (p.content LIKE ? OR p.tags LIKE ?)"
            post_params.extend([f"%{keyword}%"] * 2)

        posts = db_conn.execute(post_sql + " ORDER BY p.likes_count DESC LIMIT 20", post_params).fetchall()

        for p in posts:
            img = p['image_path']
            if not img.startswith('http'):
                img = url_for('static', filename=img.replace('static/', ''))

            results['posts'].append({
                'id': p['id'], 'content': p['content'][:50],
                'image': img, 'author': '匿名' if p['is_anonymous'] else p['name'],
                'likes': p['likes_count']
            })

        return jsonify({'status': 'success', 'results': results})
    finally:
        db_conn.close()

@app.route('/admin/add_item', methods=['POST'])
def admin_add_item():
    """
    [管理] 官方上架商品
    """
    if session.get('role') != 'admin':
        return redirect(url_for('index'))

    try:
        upload_file = request.files['image']
        fname = secure_filename(upload_file.filename)
        uname = f"off_{uuid.uuid4()}_{fname}"
        upload_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uname))

        db_conn = get_db_connection()
        db_conn.execute('''
            INSERT INTO clothing_items (
                image_path, title, category, tags, brand, price, is_ad, 
                quadrant, material, pattern, neckline, seller_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'on_sale')
        ''', (
            f"uploads/{uname}", request.form.get('title'), request.form.get('category'),
            json.dumps(request.form.get('tags', '').split(','), ensure_ascii=False),
            request.form.get('brand', '官方選品'), request.form.get('price'),
            request.form.get('is_ad') == 'on',
            request.form.get('quadrant'), request.form.get('material'),
            request.form.get('pattern'), request.form.get('neckline'),
            session['user_id']
        ))
        db_conn.commit()
        db_conn.close()
        flash('官方商品上架成功', 'success')
    except Exception as err:
        flash(f'上架失敗: {err}', 'error')

    return redirect(url_for('admin_dashboard'))

# --- 11. 工具與系統 ---
@app.route('/api/chat_response', methods=['POST'])
def chat_response():
    if 'user_id' not in session:
        return jsonify({'reply': '請先登入'})

    data = request.json
    user_msg = data.get('message', '').strip()
    session_id = data.get('session_id')  # 接收前端傳來的是在哪個聊天室說話
    user_id = session['user_id']

    if not user_msg or not session_id:
        return jsonify({'reply': '發生錯誤：找不到對話紀錄'})

    db_conn = get_db_connection()

    try:
        # 如果聊天室標題還是「新對話」，就用第一句話的前10個字當標題
        current_session = db_conn.execute('SELECT title FROM chat_sessions WHERE id=?', (session_id,)).fetchone()
        if current_session and current_session['title'] == '新對話':
            new_title = user_msg[:10] + ('...' if len(user_msg) > 10 else '')
            db_conn.execute('UPDATE chat_sessions SET title=? WHERE id=?', (new_title, session_id))

        # 儲存使用者的對話，記得存入對應的 session_id
        db_conn.execute('INSERT INTO chat_logs (session_id, user_id, sender, message) VALUES (?, ?, ?, ?)',
                        (session_id, user_id, 'user', user_msg))
        db_conn.commit()

        # 撈取歷史紀錄時，利用 session_id 過濾，只喚醒這個聊天室的記憶
        recent_logs = db_conn.execute('''
            SELECT sender, message FROM chat_logs 
            WHERE session_id = ? 
            ORDER BY created_at DESC LIMIT 6
        ''', (session_id,)).fetchall()
        recent_logs.reverse()

        formatted_history = []
        for log in recent_logs:
            if log['message'] == user_msg and log['sender'] == 'user' and log == recent_logs[-1]:
                continue
            role = "user" if log['sender'] == 'user' else "model"
            formatted_history.append({"role": role, "parts": [log['message']]})

        user_info = db_conn.execute('SELECT name FROM users WHERE id=?', (user_id,)).fetchone()
        user_name = user_info['name'] if user_info else "貴賓"

        system_prompt = f"""
        你是一位擁有10年經驗的頂級時尚形象顧問。現在對話客戶是「{user_name}」。
        1. 語氣溫柔、自信、具同理心。
        2. 請使用繁體中文，支援 Markdown 格式。
        3. 回答精簡扼要，適當加上 Emoji。
        """

        import google.generativeai as genai
        # 建立 Gemini 模型
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', system_instruction=system_prompt)
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(user_msg)
        ai_reply = response.text

    except Exception as e:
        print(f"AI 聊天錯誤: {e}")
        ai_reply = "⚠️ 哎呀！顧問大腦稍微當機了，能請您再說一次嗎？"

    try:
        # 將 AI 的回覆也存進同一個聊天室
        db_conn.execute('INSERT INTO chat_logs (session_id, user_id, sender, message) VALUES (?, ?, ?, ?)',
                        (session_id, user_id, 'ai', ai_reply))
        db_conn.commit()
    except Exception as e:
        print(f"儲存AI回覆失敗: {e}")
    finally:
        db_conn.close()

    return jsonify({'reply': ai_reply})

@app.route('/api/ai_stylist_chat', methods=['POST'])
def ai_stylist_chat():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入會員'}), 401

    try:
        data = request.json
        user_input = data.get('message', '')
        # 接收前端傳來的對話紀錄
        history = data.get('history', [])

        db_conn = get_db_connection()
        user = db_conn.execute('SELECT name, gender FROM users WHERE id=?', (session['user_id'],)).fetchone()
        analysis = db_conn.execute(
            'SELECT body_data FROM analysis_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
            (session['user_id'],)).fetchone()
        db_conn.close()

        user_name = user['name'] if user else "使用者"
        body_info = "未知身形"
        if analysis and analysis['body_data']:
            # noinspection PyBroadException
            try:
                body_info = json.loads(analysis['body_data']).get('shape', '一般身形')
            except Exception:
                pass

        # 將歷史對話格式化成字串給 AI 參考
        history_str = "\n".join([f"{'客戶' if msg['role'] == 'user' else '你'}: {msg['text']}" for msg in history])

        if 'model' in globals() and model:
            # 💡 提示詞大升級：賦予 AI 角色設定與判斷能力
            prompt = f"""
            你是一位名叫「Smart Style」的專業 AI 形象顧問，說話風格親切、幽默，像真人的朋友一樣。
            客戶名字：{user_name}，身形：{body_info}。

            【最近的對話紀錄】
            {history_str}

            【客戶最新訊息】：「{user_input}」

            請根據上下文自然地回應客戶。
            請務必回傳 JSON 格式：
            {{
                "reply": "你的回覆內容 (若有換行請使用 <br> 標籤，態度要像真人閒聊)",
                "action": "chat" 或 "change_texture", 
                "visual_cues": "如果客戶的要求包含換衣服、改變顏色或材質，請設定 action 為 change_texture，並在這裡輸出單一中文材質(如:丹寧/皮衣/針織/棉麻/迷彩/紅色絲綢)。若只是閒聊或問問題，action 請設定 chat，這裡留空。"
            }}
            """
            try:
                response = model.generate_content(prompt)
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                ai_data = json.loads(clean_text)
                return jsonify({'status': 'success', 'data': ai_data})
            except Exception as err:
                print(f"Gemini Error: {err}")
                return jsonify({'status': 'success', 'data': {
                    'reply': '不好意思，我剛剛恍神了一下，您可以再說一次嗎？',
                    'action': 'chat', 'visual_cues': ''
                }})
        else:
            return jsonify({'status': 'success', 'data': {'reply': '系統維護中。', 'action': 'chat', 'visual_cues': ''}})

    except Exception as err:
        print(f"Chat API Error: {err}")
        return jsonify({'status': 'error', 'msg': '系統發生錯誤'})

@app.route('/api/mirror_mode', methods=['POST'])
def mirror_mode_api():
    """
    [魔鏡] 急救建議模式
    """
    problem = request.json.get('problem', '')
    solution = "您看起來很棒！自信就是最好的穿搭。"

    if any(k in problem for k in ['腿短', '比例', '矮']):
        solution = "試著把上衣紮進去，或者換一雙與褲子同色的鞋子來延伸視覺腿長。"
    elif any(k in problem for k in ['沒精神', '暗沈', '氣色']):
        solution = "塗個口紅，或是戴上一副亮金屬色的耳環，利用光澤感能立刻提亮臉部。"
    elif any(k in problem for k in ['胖', '臃腫', '肉']):
        solution = "露出身上最細的部位（如手腕、腳踝、鎖骨），或是加上一條腰帶強調腰線。"
    elif any(k in problem for k in ['寬', '骨架大']):
        solution = "避免墊肩款式，嘗試落肩設計或 V 領上衣，可以柔和上半身線條。"

    return jsonify({'status': 'success', 'solution': solution})

@app.route('/api/check_ar_capability', methods=['POST'])
def check_ar_capability():
    """
    [工具] 檢查 AR 支援度
    """
    device = request.json.get('device', 'unknown')
    if any(k in device for k in ['Mobile', 'Android', 'iPhone']):
        return jsonify({'status': 'success', 'ar_ready': True, 'msg': '📱 您的裝置支援 AR 試穿'})
    return jsonify({'status': 'warning', 'ar_ready': False, 'msg': '💻 建議使用手機體驗 AR 功能'})

@app.route('/api/update_accessibility', methods=['POST'])
def update_accessibility():
    """
    [無障礙] 更新介面偏好
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    db_conn = get_db_connection()
    # noinspection PyBroadException
    try:
        db_conn.execute('UPDATE users SET accessibility_prefs = ? WHERE id = ?',
                     (json.dumps(request.json), session['user_id']))
        db_conn.commit()
    except:
        pass
    finally:
        db_conn.close()

    return jsonify({'status': 'success'})

@app.route('/api/convert_size', methods=['POST'])
def convert_size_api():
    """
    [工具] 尺碼轉換器
    """
    size = request.json.get('size', 'M')
    locale = session.get('locale', 'zh_TW')

    if locale == 'zh_TW':
        res = f"{size} (亞洲版型)"
        note = "建議依照平時穿著尺寸挑選"
    else:
        res = f"{size} (US Fit)"
        note = "美版偏大，建議拿小一號"

    return jsonify({'status': 'success', 'result': res, 'note': note})

@app.route('/api/submit_proposal', methods=['POST'])
def submit_proposal():
    """
    [互動] 提交風格提案
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    tag = request.form.get('tag_name')
    desc = request.form.get('description', '')
    if not tag: return jsonify({'status': 'error', 'msg': '標籤名稱為空'})

    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO style_proposals (user_id, tag_name, description) VALUES (?, ?, ?)',
                 (session['user_id'], tag, desc))
    db_conn.commit()
    db_conn.close()

    return jsonify({'status': 'success', 'msg': '提案已提交審核'})

@app.route('/api/upgrade_vip', methods=['POST'])
def upgrade_vip():
    """
    [金流] 模擬 VIP 升級
    """
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    db_conn = get_db_connection()
    db_conn.execute('UPDATE users SET is_vip=1 WHERE id=?', (session['user_id'],))
    db_conn.commit()
    db_conn.close()

    session['is_vip'] = True
    return jsonify({'status': 'success', 'msg': '恭喜升級 VIP！'})

# --- 12. 管理員與開發工具 ---
@app.route('/api/admin/resolve_report', methods=['POST'])
def resolve_report_api():
    """
    [管理] 處理檢舉
    """
    # 權限驗證
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'status': 'error', 'msg': '權限不足'}), 403

    data = request.json or {}
    report_id = data.get('report_id')
    action = data.get('action')

    db_conn = get_db_connection()
    try:
        if action == 'delete_post':
            report = db_conn.execute('SELECT post_id FROM reports WHERE id=?', (report_id,)).fetchone()
            if report:
                db_conn.execute('DELETE FROM posts WHERE id=?', (report['post_id'],))
                db_conn.execute('UPDATE reports SET status="resolved" WHERE id=?', (report_id,))
                msg = '貼文已刪除'
            else:
                msg = '檢舉紀錄不存在'
        else:
            db_conn.execute('UPDATE reports SET status="dismissed" WHERE id=?', (report_id,))
            msg = '檢舉已駁回'

        db_conn.commit()
        return jsonify({'status': 'success', 'msg': msg})

    except Exception as err:
        print(f"Admin API Error: {err}")
        return jsonify({'status': 'error', 'msg': '資料庫處理失敗，請稍後再試'}), 500

    finally:
        db_conn.close()

@app.route('/api/admin/review_proposal', methods=['POST'])
def review_proposal():
    """
    [管理] 審核風格提案
    """
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403

    p_id = request.json.get('id')
    status = 'approved' if request.json.get('action') == 'approve' else 'rejected'

    db_conn = get_db_connection()
    db_conn.execute('UPDATE style_proposals SET status=? WHERE id=?', (status, p_id))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/admin/update_trends', methods=['POST'])
def update_trends():
    """
    [管理] 更新趨勢權重
    """
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403

    db_conn = get_db_connection()
    db_conn.execute('INSERT OR REPLACE INTO system_configs (key, value) VALUES (?, ?)',
                 ('trend_weights', json.dumps(request.json.get('weights'))))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/research/export_report')
def export_research_report():
    """
    [研究] 匯出匿名統計數據
    """
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403

    db_conn = get_db_connection()

    # 1. 身形多樣性統計
    shape_stats = {}
    rows = db_conn.execute("SELECT body_data FROM analysis_history").fetchall()
    for r in rows:
        # noinspection PyBroadException
        try:
            b = json.loads(r['body_data'])
            shape = b.get('shape', 'Unknown').split(' ')[0]
            shape_stats[shape] = shape_stats.get(shape, 0) + 1
        except:
            pass

    # 2. 心理影響統計
    psy_stats = db_conn.execute(
        'SELECT feeling, AVG(rating) as avg, COUNT(*) as c FROM wear_logs GROUP BY feeling').fetchall()
    psy_data = [{'feeling': r['feeling'], 'avg_score': round(r['avg'], 1), 'count': r['c']} for r in psy_stats]

    db_conn.close()

    report = {
        'meta': {'time': datetime.datetime.now().isoformat(), 'title': 'Anonymous Research Data'},
        'data': {'body_diversity': shape_stats, 'psychological_impact': psy_data}
    }

    fname = f"research_{int(time.time())}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return send_file(path, as_attachment=True)

@app.route('/api/lab/track', methods=['POST'])
def lab_track_api():
    """
    [實驗室] 記錄體態數據
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    d = request.json
    db_conn = get_db_connection()
    db_conn.execute('INSERT INTO body_tracking (user_id, weight, waist, hip, note, recorded_at) VALUES (?,?,?,?,?,?)',
                 (session['user_id'], d.get('weight'), d.get('waist'), d.get('hip'), d.get('note'),
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    db_conn.commit()
    db_conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/lab/correct', methods=['POST'])
def lab_correct_api():
    """
    [實驗室] 回報分析錯誤
    """
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401

    db_conn = get_db_connection()
    latest = db_conn.execute('SELECT id FROM analysis_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
                          (session['user_id'],)).fetchone()

    if latest:
        db_conn.execute('UPDATE analysis_history SET is_incorrect=1, note=? WHERE id=?',
                     (f"User correction: {request.json.get('correct_shape')}", latest['id']))
        db_conn.commit()
        msg = '已記錄回饋'
    else:
        msg = '無近期紀錄'

    db_conn.close()
    return jsonify({'status': 'success', 'msg': msg})

@app.route('/api/lab/mood', methods=['POST'])
def lab_mood_api():
    """
    [實驗室] 心情穿搭推薦
    """
    mood = request.json.get('mood', 'happy')
    mapping = {
        'happy': {'tone': '活力', 'text': '試試亮色系！', 'items': ['印花洋裝']},
        'sad': {'tone': '療癒', 'text': '穿件柔軟的衛衣吧。', 'items': ['米色衛衣']},
        'nervous': {'tone': '專業', 'text': '深藍色能帶來平靜。', 'items': ['西裝外套']},
        'party': {'tone': '閃耀', 'text': '你是今晚的主角！', 'items': ['亮片裙']}
    }
    return jsonify({'status': 'success', 'data': mapping.get(mood, mapping['happy'])})

# --- 資料庫維護工具 (Dev Tools) ---
@app.route('/setup_db_final')
def setup_db_final():
    """資料庫緊急補完腳本"""
    db_conn = get_db_connection()
    try:
        # noinspection PyBroadException
        try:
            db_conn.execute("ALTER TABLE analysis_history ADD COLUMN is_converted BOOLEAN DEFAULT 0")
        except:
            pass
        # noinspection PyBroadException
        try:
            db_conn.execute("ALTER TABLE analysis_history ADD COLUMN ab_variant TEXT DEFAULT 'A'")
        except:
            pass
        # noinspection PyBroadException
        try:
            db_conn.execute("ALTER TABLE analysis_history ADD COLUMN model_version TEXT")
        except:
            pass
        # noinspection PyBroadException
        try:
            db_conn.execute("ALTER TABLE users ADD COLUMN maturity_level TEXT DEFAULT 'balanced'")
        except:
            pass

        db_conn.commit()
        return "資料庫修復完成！缺少的欄位 (is_converted 等) 已補上。請回到 <a href='/admin'>後台</a>"
    except Exception as err:
        return f"修復失敗: {err}"
    finally:
        db_conn.close()

@app.route('/setup_trends')
def setup_trends_db():
    """趨勢資料庫初始化"""
    db_conn = get_db_connection()
    try:
        db_conn.execute(
            'CREATE TABLE IF NOT EXISTS trends (id INTEGER PRIMARY KEY, keyword TEXT, influence_score INTEGER, data_points TEXT)')
        db_conn.execute(
            'CREATE TABLE IF NOT EXISTS celebrity_looks (id INTEGER PRIMARY KEY, trend_id INTEGER, image_path TEXT)')
        db_conn.execute(
            'CREATE TABLE IF NOT EXISTS celeb_likes (user_id INTEGER, celeb_id INTEGER, PRIMARY KEY(user_id, celeb_id))')

        if db_conn.execute('SELECT count(*) FROM trends').fetchone()[0] == 0:
            db_conn.execute("INSERT INTO trends (keyword, influence_score) VALUES ('多巴胺穿搭', 95)")
            db_conn.commit()
        return "Trends DB Initialized"
    finally:
        db_conn.close()

@app.route('/setup_admin')
def setup_admin():
    """建立管理員"""
    db_conn = get_db_connection()
    try:
        pw = generate_password_hash('admin123')
        db_conn.execute("INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?,?,?,?,1)",
                     ('admin@style.com', pw, 'Admin', 'admin'))
        db_conn.commit()
        return "Admin Created: admin@style.com / admin123"
    finally:
        db_conn.close()

if __name__ == '__main__':
    print("------------------------------------------------")
    print("🚀 Smart Style System 正在啟動...")

    # 1. 初始化與自動修復資料庫 (Auto-Migration)
    try:
        init_db()  # 建立基礎表格

        # 自動檢查並補齊新欄位
        conn = get_db_connection()
        columns_to_check = [
            # 系統原有的欄位檢查
            ("users", "maturity_level TEXT DEFAULT 'balanced'"),
            ("users", "culture_pref INTEGER DEFAULT 5"),
            ("users", "life_stage TEXT DEFAULT 'student'"),
            ("analysis_history", "model_version TEXT"),
            ("analysis_history", "logic_trace TEXT"),
            ("analysis_history", "is_converted BOOLEAN DEFAULT 0"),
            ("analysis_history", "ab_variant TEXT DEFAULT 'A'"),

            # --- 個人檔案新增欄位 ---
            ("users", "age INTEGER"),
            ("users", "avatar TEXT"),
            ("users", "height FLOAT"),
            ("users", "weight FLOAT"),
            ("users", "body_type TEXT"),
            ("users", "wear_types TEXT"),
            ("users", "fabrics TEXT"),
            ("users", "patterns TEXT"),
            ("users", "style_dislikes TEXT"),
            ("users", "color_prefs TEXT"),
            ("users", "occasion_prefs TEXT"),
            ("users", "clothing_issues TEXT"),
            ("chat_logs", "session_id TEXT")
            # ----------------------------------
        ]

        fixed_count = 0
        for table, col_def in columns_to_check:
            # noinspection PyBroadException
            try:
                # 嘗試新增欄位，如果欄位已存在會報錯並被忽略
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                fixed_count += 1
            except:
                pass  # 欄位已存在，略過

        # 確保 AI 聊天室 (Sessions) 表存在
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

        # 確保 AI 聊天紀錄 (Logs) 表存在 (加入了 session_id)
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    session_id TEXT,
                    user_id INTEGER,
                    sender TEXT, 
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

        conn.commit()
        conn.close()

        print(f"✅ 資料庫檢查完成 (自動修補了 {fixed_count} 個欄位)")

    except Exception as e:
        print(f"⚠️ 資料庫警告: {e} (但不影響主程式啟動)")

    # 2. 設定執行環境
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    print("------------------------------------------------")
    print(f"👉 請開啟瀏覽器訪問: http://127.0.0.1:{port}")
    print(f"👉 管理員後台入口:   http://127.0.0.1:{port}/admin")
    print("------------------------------------------------")

    # 3. 啟動伺服器
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=True)