import os
import sqlite3
import uuid
import json
import random
import datetime
import shutil
import re
import time
import cv2
import mediapipe as mp
import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib.parse
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from fpdf import FPDF
from gradio_client import Client, file  # 確保有安裝 gradio_client

app = Flask(__name__)
app.secret_key = 'thesis_final_ultimate_v2026_complete_edition'

# --- 15. 系統全域設定 (模型版本與 API Key) ---
CURRENT_MODEL_VERSION = "StyleNet-Evo-v3.0"  # 升級為 v3.0 演化版
API_ACCESS_KEY = "open_style_api_2026"

# 設定圖片上傳路徑
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# --- 倫理與安全設定 (Ethical Guardrails & Positive Marketing) ---
class ContentSafety:
    # 12. 不以缺點行銷：負面詞彙轉正向字典
    NEGATIVE_WORDS = {
        '胖': '豐滿圓潤', '肥': '棉花糖系', '粗': '線條明顯', '短': '嬌小',
        '大餅臉': '圓潤臉型', '五五身': '腰線待調整', '象腿': '腿部線條明顯',
        '平胸': '纖細骨感', '虎背熊腰': '上身較為厚實', '水桶腰': '直筒身形',
        '蘿蔔腿': '小腿肌明顯',
        '醜': '具個人特色', '難看': '有進步空間', '老氣': '復古成熟',
        '糟糕': '待優化', '奇怪': '前衛', '顯胖': '視覺膨脹感'
    }

    DISTRESS_KEYWORDS = ['想死', '自殺', '沒用', '討厭自己', '絕望', '痛苦']
    DISCLAIMER = "【溫馨提醒】美的標準由您定義。本系統建議僅供穿搭參考，希望能為您的自信加分。"

    @staticmethod
    def sanitize(text):
        """將所有負面描述轉化為建設性建議"""
        if not text: return text
        for bad, good in ContentSafety.NEGATIVE_WORDS.items():
            text = text.replace(bad, good)
        return text

    @staticmethod
    def check_mental_health(text):
        if not text: return False
        return any(k in text for k in ContentSafety.DISTRESS_KEYWORDS)

# --- [更新] 全方位風格定位矩陣 (含髮型/妝容/配件) ---
STYLE_MATRIX = {
    "Q1": {  # 年輕 x 柔和
        "name": "清純甜美 / 溫暖鄰家系",
        "keywords": ["靈動", "親切", "乾淨", "柔軟"],
        "archetype": {
            "female": "少女型 (Sweet/Ingenue)",
            "male": "陽光小奶狗 / 溫柔弟弟"
        },
        "clothing_guide": "短款、鮮豔色、小圖案、輕薄材質、針織、棉麻、圓領、荷葉邊。",
        "hairstyle": "空氣瀏海、丸子頭、羊毛捲、半紮髮",
        "makeup": "偽素顏、粉嫩腮紅、水光嘟嘟唇、臥蠶",
        "accessories": "蝴蝶結髮飾、細項鍊、珍珠耳釘、帆布包"
    },
    "Q2": {  # 年輕 x 硬朗
        "name": "少年叛逆 / 潮流前衛系",
        "keywords": ["俐落", "率真", "個性", "機靈"],
        "archetype": {
            "female": "少年型 / 前衛型 (Gamine/Avant-Garde)",
            "male": "痞帥小狼狗 / 酷蓋 (Cool Guy)"
        },
        "clothing_guide": "短款、幾何剪裁、工裝、牛仔、皮衣、不規則設計、對比色。",
        "hairstyle": "高層次短髮、狼尾頭、挑染、俐落直髮",
        "makeup": "小煙燻、個性眼線、霧面唇彩、立體眉型",
        "accessories": "金屬耳骨夾、頸鍊 (Choker)、棒球帽、銀飾"
    },
    "Q3": {  # 成熟 x 柔和
        "name": "優雅浪漫 / 貴族氣質系",
        "keywords": ["高級", "精緻", "華麗", "深情"],
        "archetype": {
            "female": "優雅型 / 浪漫型 (Elegant/Romantic)",
            "male": "儒雅紳士 / 混血貴公子"
        },
        "clothing_guide": "長款(風衣/大衣)、絲質襯衫、高級西裝、柔軟垂墜材質、大氣印花。",
        "hairstyle": "大波浪捲髮、側分長瀏海、低盤髮、法式慵懶捲",
        "makeup": "經典紅唇、精緻眼影、光澤底妝、修容",
        "accessories": "絲巾、垂墜耳環、精品手錶、手拿包"
    },
    "Q4": {  # 成熟 x 硬朗
        "name": "霸氣權威 / 經典職場系",
        "keywords": ["嚴謹", "端莊", "強勢", "幹練"],
        "archetype": {
            "female": "古典型 / 戲劇型 (Classic/Dramatic)",
            "male": "商業大亨 / 霸氣硬漢"
        },
        "clothing_guide": "正裝筆挺、硬挺材質、深色、無圖案或大幾何、西裝領、直線條剪裁。",
        "hairstyle": "俐落中分、大背頭、直長髮、低馬尾",
        "makeup": "大地色眼妝、俐落眉峰、裸色或深紅唇膏",
        "accessories": "幾何金屬飾品、胸針、皮帶、大托特包"
    },
    "CENTER": {  # 平衡
        "name": "自然隨性 / 舒適休閒系",
        "keywords": ["隨意", "大方", "舒適", "耐看"],
        "archetype": {
            "female": "自然型 (Natural)",
            "male": "爹系男友 / 鄰家哥哥"
        },
        "clothing_guide": "基礎款、棉麻材質、寬鬆舒適、無印良品風、大地色系。",
        "hairstyle": "鎖骨髮、微捲中長髮、自然直髮、高馬尾",
        "makeup": "清透底妝、野生眉、豆沙色口紅",
        "accessories": "簡約手環、草編帽、托特包、細框眼鏡"
    }
}

# --- [新增] 全域風格定義 (Style Taxonomy) ---
STYLE_TAXONOMY = {
    "region": ["中國風", "日系", "韓系", "歐美", "英倫", "法式", "波西米亞"],
    "scene": ["通勤", "休閒", "田園", "校園", "運動", "Party", "約會", "度假"],
    "design": ["新中式", "淑女", "名媛", "瑞麗", "簡約", "極簡", "中性", "性冷淡", "民族", "戲劇", "復古"],
    "trend": ["Y2K", "嘻哈", "朋克", "嘻皮", "甜酷"],
    "art": ["拜占庭", "浪漫", "哥特", "巴洛克", "洛可可", "洛麗塔", "維多利亞", "新古典", "超現實", "波普藝術", "歐普",
            "未來色彩", "多巴胺", "美拉德"]
}

# 輔助函式：將所有標籤攤平成一個列表 (供搜尋用)
ALL_STYLE_TAGS = [tag for category in STYLE_TAXONOMY.values() for tag in category]

# --- 多語系設定 ---
TRANSLATIONS = {'zh_TW': {'currency': 'NT$'}, 'en_US': {'currency': 'US$'}}


# --- 資料庫初始化 ---
def get_db_connection():
    conn = sqlite3.connect('style_system.db')
    conn.row_factory = sqlite3.Row
    return conn


# --- [優化版] 真實 AI 影像處理引擎 (Real AI Engine) ---
class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # [優化] 關閉 refine_landmarks 以提升速度
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )

    def analyze(self, image_path):
        """傳入圖片路徑，回傳真實的臉型與五官數據"""
        image = cv2.imread(image_path)
        if image is None:
            return None, "無法讀取圖片"

        # [新增] 圖片尺寸優化
        h, w = image.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_w = 1280
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None, "未偵測到臉部，請確保照片清晰且包含完整臉部"

        landmarks = results.multi_face_landmarks[0].landmark

        def get_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        cheek_width = np.linalg.norm(get_pt(454) - get_pt(234))
        jaw_width = np.linalg.norm(get_pt(132) - get_pt(361))
        face_length = np.linalg.norm(get_pt(10) - get_pt(152))
        forehead_width = np.linalg.norm(get_pt(338) - get_pt(109))

        ratio_len_width = face_length / cheek_width
        ratio_jaw_cheek = jaw_width / cheek_width

        shape = "未知"
        if ratio_len_width > 1.45:
            shape = "長型臉 (Oblong)"
        elif ratio_len_width < 1.15 and ratio_jaw_cheek > 0.9:
            shape = "方型臉 (Square)"
        elif ratio_len_width < 1.15:
            shape = "圓型臉 (Round)"
        elif forehead_width < cheek_width and jaw_width < cheek_width * 0.7:
            shape = "鑽石臉 (Diamond)"
        elif jaw_width < cheek_width * 0.6 and forehead_width > jaw_width:
            shape = "心型臉 (Heart)"
        elif ratio_len_width > 1.25 and ratio_len_width <= 1.45:
            if ratio_jaw_cheek > 0.8:
                shape = "長方臉 (Rectangle)"
            else:
                shape = "鵝蛋臉 (Oval)"
        else:
            shape = "鵝蛋臉 (Oval)"

        left_eye_width = np.linalg.norm(get_pt(33) - get_pt(133))
        eye_dist = np.linalg.norm(get_pt(133) - get_pt(362))

        eye_feature = "標準眼距"
        if eye_dist > left_eye_width * 1.3:
            eye_feature = "眼距較寬"
        elif eye_dist < left_eye_width * 0.9:
            eye_feature = "眼距較近"

        return {
            'shape': shape,
            'ratios': {
                'face_ratio (L/W)': round(ratio_len_width, 2),
                'jaw_cheek_ratio': round(ratio_jaw_cheek, 2)
            },
            'features': {
                'eyes': eye_feature,
                'cheekbones': '明顯' if cheek_width > forehead_width * 1.1 else '柔和',
                'jawline': '稜角分明' if shape in ['方型臉', '長方臉'] else '圓潤'
            }
        }, None


face_engine = FaceAnalyzer()


# --- 真實 AI 身材分析引擎 (Body Analysis Engine) ---
class BodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None: return None, "無法讀取圖片"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, "未偵測到全身或半身特徵"

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        def get_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        shoulder_width = np.linalg.norm(get_pt(11) - get_pt(12))
        hip_width = np.linalg.norm(get_pt(23) - get_pt(24))
        if hip_width == 0: hip_width = 0.001

        sh_ratio = shoulder_width / hip_width
        body_shape = "勻稱型"
        desc = ""

        if sh_ratio > 1.05:
            body_shape = "倒三角型 (Inverted Triangle)"
            desc = "肩寬明顯大於臀寬，建議穿著修飾下身的單品。"
        elif sh_ratio < 0.92:
            body_shape = "梨型 (Pear)"
            desc = "臀寬明顯大於肩寬，建議強調上半身線條。"
        else:
            body_shape = "矩形/沙漏型 (Rectangle/Hourglass)"
            desc = "肩臀比例平衡，適合強調腰線。"

        proportion_info = "全身比例數據不足"
        try:
            shoulder_mid = (get_pt(11) + get_pt(12)) / 2
            hip_mid = (get_pt(23) + get_pt(24)) / 2
            if landmarks[29].visibility > 0.5 and landmarks[30].visibility > 0.5:
                heel_mid = (get_pt(29) + get_pt(30)) / 2
                torso_len = np.linalg.norm(shoulder_mid - hip_mid)
                leg_len = np.linalg.norm(hip_mid - heel_mid)
                if leg_len > torso_len * 1.4:
                    proportion_info = "長腿比例"
                else:
                    proportion_info = "標準比例"
        except:
            pass

        try:
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            if landmarks[29].visibility > 0.5:
                ankle_y = (landmarks[29].y + landmarks[30].y) / 2
            else:
                ankle_y = 0.95
        except:
            shoulder_y, hip_y, ankle_y = 0.2, 0.5, 0.9

        return {
            'shape': body_shape,
            'ratios': {'shoulder_hip_ratio': round(sh_ratio, 2), 'proportion_type': proportion_info},
            'advice': desc,
            'landmarks': {'shoulder_y': shoulder_y, 'hip_y': hip_y, 'ankle_y': ankle_y}
        }, None


body_engine = BodyAnalyzer()


# --- 外部電商模擬適配器 (修復版) ---
class ExternalShopAdapter:
    def __init__(self):
        self.partners = [
            {'name': 'Uniqlo', 'url_base': 'https://www.uniqlo.com/tw/'},
            {'name': 'GU', 'url_base': 'https://www.gu-global.com/tw/'},
            {'name': 'ZARA', 'url_base': 'https://www.zara.com/tw/'}
        ]

    def fetch_items(self, category=None, min_price=0, max_price=10000):
        external_items = []
        count = random.randint(3, 5)
        adjectives = ['當季', '熱銷', '聯名', '修身', '透氣']
        nouns = {'top': ['T恤', '襯衫', '針織衫'], 'bottom': ['寬褲', '牛仔褲', '長裙'], 'outer': ['外套', '大衣']}
        target_nouns = nouns.get(category, nouns['top'] + nouns['bottom'])

        for _ in range(count):
            partner = random.choice(self.partners)
            price = random.randint(max(190, int(min_price)), min(2990, int(max_price)))
            title = f"{partner['name']} {random.choice(adjectives)} {random.choice(target_nouns)}"
            item = {
                'id': f"ext_{uuid.uuid4()}",
                'title': title,
                'brand': partner['name'],
                'price': price,
                'image': 'https://placehold.co/400x300?text=Fashion+Item',
                'is_external': True,
                'link': partner['url_base'],
                'trust_score': random.randint(80, 99)
            }
            external_items.append(item)
        return external_items


shop_adapter = ExternalShopAdapter()


# --- 真實 AI 虛擬試穿引擎 ---
class TryOnEngine:
    def __init__(self):
        self.client_url = "yisol/IDM-VTON"
        self.client = None

    def initialize(self):
        if not self.client:
            print("正在連接雲端試穿模型 IDM-VTON...")
            try:
                self.client = Client(self.client_url)
                print("雲端模型連接成功！")
            except Exception as e:
                print(f"雲端模型連接失敗: {e}")

    def generate(self, person_img_path, garment_img_path, category="upper_body"):
        self.initialize()
        if not self.client:
            return None, "無法連接 AI 伺服器"
        try:
            abs_person = os.path.abspath(person_img_path)
            abs_garment = os.path.abspath(garment_img_path)
            result = self.client.predict(
                dict={"background": file(abs_person), "layers": [], "composite": None},
                garm_img=file(abs_garment),
                garment_des="clothing",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            return result[0], None
        except Exception as e:
            print(f"VTON 生成失敗: {e}")
            return None, str(e)


vton_engine = TryOnEngine()


# --- 真實 AI 投票分析引擎 ---
class VoteInsightEngine:
    def __init__(self):
        self.client_url = "vikhyatk/moondream2"
        self.client = None

    def initialize(self):
        if not self.client:
            print("正在連接視覺分析模型 Moondream2...")
            try:
                self.client = Client(self.client_url)
                print("視覺模型連接成功！")
            except Exception as e:
                print(f"視覺模型連接失敗: {e}")

    def analyze(self, image_path, vote_result):
        self.initialize()
        if not self.client:
            return "系統忙碌中，無法進行視覺分析。"
        if vote_result == 'popular':
            prompt = "Describe why this outfit is stylish and looks good. Keep it brief."
        else:
            prompt = "Describe what could be improved in this outfit. Keep it brief."
        try:
            abs_path = os.path.abspath(image_path)
            result = self.client.predict(file(abs_path), prompt, api_name="/answer_question")
            return f"AI 視覺分析：{result}"
        except Exception as e:
            print(f"投票分析失敗: {e}")
            return "AI 正在學習這種風格，暫時無法評論。"


vote_engine = VoteInsightEngine()


# --- [新增] AI 網路趨勢搜查員 ---
def fetch_trends_from_web():
    """
    連動網路：自動去 Google News 搜尋最新的時尚趨勢
    """
    print("🌐 正在連線網路，搜尋最新流行趨勢...")

    # 搜尋關鍵字
    search_query = "2024 流行穿搭 風格 趨勢"
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://www.google.com/search?q={encoded_query}&tbm=nws&hl=zh-TW&gl=TW"

    # 偽裝成一般瀏覽器 (避免被 Google 阻擋)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    live_trends = []

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 抓取新聞標題區塊 (Google 結構可能會變，這是目前通用的抓法)
            articles = soup.find_all('div', class_='SoaBEf')

            for article in articles[:6]:  # 只抓前 6 條最新新聞
                try:
                    # 抓標題
                    title_div = article.find('div', role='heading')
                    title = title_div.text if title_div else "新一季流行趨勢"

                    # 抓摘要
                    desc_div = article.find('div', class_='GI74Re')
                    desc = desc_div.text if desc_div else "點擊查看更多細節..."

                    # 簡單的關鍵字提取 (AI 模擬)
                    keyword = title[:10]  # 取標題前幾個字當關鍵字
                    if "色" in title:
                        category = "Art/Color"
                    elif "風格" in title or "風" in title:
                        category = "Style"
                    elif "鞋" in title or "包" in title:
                        category = "Item"
                    else:
                        category = "Trend"

                    # 隨機生成熱度數據
                    score = random.randint(80, 99)
                    points = sorted([random.randint(60, 100) for _ in range(5)])

                    live_trends.append((keyword, category, score, desc, json.dumps(points)))
                    print(f"✅ 抓取成功: {title}")
                except:
                    continue
        else:
            print(f"❌ 無法連線至 Google News (Status: {response.status_code})")

    except Exception as e:
        print(f"⚠️ 網路爬蟲發生錯誤: {e}")
        # 如果沒網路或報錯，回傳空清單，避免程式崩潰
        return []

    return live_trends


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    print("🚀 正在檢查並初始化資料庫 (完整欄位版)...")

    # ==========================================
    # 1. 建立所有資料表 (Create Tables)
    # ==========================================

    # --- 核心用戶與分析 ---
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, name TEXT NOT NULL, role TEXT DEFAULT 'user', status TEXT DEFAULT 'active', is_vip BOOLEAN DEFAULT 0, data_consent BOOLEAN DEFAULT 0, tos_version TEXT DEFAULT '1.0', locale TEXT DEFAULT 'zh_TW', age INTEGER, gender TEXT, height REAL, weight REAL, maturity_level TEXT DEFAULT 'balanced', culture_pref INTEGER DEFAULT 5, life_stage TEXT DEFAULT 'student', clothing_issues TEXT, style_preferences TEXT, color_preferences TEXT, occasion_preferences TEXT, photo_policy TEXT DEFAULT '30_days', ai_training_consent BOOLEAN DEFAULT 0, accessibility_prefs TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS analysis_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, user_image_path TEXT, face_data TEXT, body_data TEXT, final_recommendation TEXT, ai_confidence INTEGER DEFAULT 85, is_incorrect BOOLEAN DEFAULT 0, user_feedback TEXT, ab_variant TEXT DEFAULT 'A', is_converted BOOLEAN DEFAULT 0, model_version TEXT, logic_trace TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS wear_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, date_str TEXT, outfit_desc TEXT, feeling TEXT, rating INTEGER, ai_adjustment_note TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS clothing_items (id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT, title TEXT, category TEXT, tags TEXT, brand TEXT, price INTEGER, is_ad BOOLEAN DEFAULT 0, trust_score INTEGER DEFAULT 95, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS user_milestones (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, milestone_type TEXT, achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))''')

    # --- 社群與互動 (修正：明確定義欄位) ---
    c.execute(
        '''CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, image_path TEXT, content TEXT, tags TEXT, is_anonymous BOOLEAN, is_qa BOOLEAN DEFAULT 0, poll_yes INTEGER DEFAULT 0, poll_no INTEGER DEFAULT 0, likes_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (id))''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, post_id INTEGER, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS likes (user_id INTEGER, post_id INTEGER, PRIMARY KEY (user_id, post_id))''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS follows (follower_id INTEGER, followed_id INTEGER, PRIMARY KEY (follower_id, followed_id))''')

    # [修正重點] Reports 表格補上 status, reason, reporter_id
    c.execute(
        '''CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, reporter_id INTEGER, post_id INTEGER, reason TEXT, status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # --- 功能模組 (修正：明確定義欄位) ---
    # [修正重點] Body Tracking 補上 weight, waist, hip
    c.execute(
        '''CREATE TABLE IF NOT EXISTS body_tracking (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, weight REAL, waist REAL, hip REAL, note TEXT, recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # [修正重點] Calendar 補上 date_str, title
    c.execute(
        '''CREATE TABLE IF NOT EXISTS calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, date_str TEXT, title TEXT, outfit_desc TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # [修正重點] Chat Logs 補上 sender, message
    c.execute(
        '''CREATE TABLE IF NOT EXISTS chat_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, sender TEXT, message TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute(
        '''CREATE TABLE IF NOT EXISTS try_on_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, original_img TEXT, cloth_img TEXT, result_img TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS favorites (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, item_data TEXT, saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # --- 趨勢與設定 ---
    c.execute(
        '''CREATE TABLE IF NOT EXISTS trends (id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT UNIQUE, category TEXT, status TEXT DEFAULT 'rising', influence_score INTEGER, description TEXT, data_points TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS celebrity_looks (id INTEGER PRIMARY KEY AUTOINCREMENT, trend_id INTEGER, celeb_name TEXT, image_path TEXT, description TEXT, FOREIGN KEY (trend_id) REFERENCES trends (id), UNIQUE(trend_id, celeb_name))''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS style_proposals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, tag_name TEXT, description TEXT, status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS system_configs (key TEXT PRIMARY KEY, value TEXT)''')

    # ==========================================
    # 2. 寫入預設資料 (初始化趨勢、用戶、商品)
    # ----------------------------------------

    # [A] 基礎趨勢資料
    static_trends = [
        ('中國風', 'Region', 85, '運用刺繡、龍鳳圖騰與絲綢面料，展現東方傳統美學的經典風格。', '[70, 75, 80, 82, 85]'),
        ('日系', 'Region', 92, '強調多層次疊穿與自然材質，色調柔和，展現舒適且細膩的日常美感。', '[85, 88, 90, 91, 92]'),
        ('韓系', 'Region', 94, '剪裁俐落修身，善用西裝外套與高腰單品，展現都會摩登的時尚感。', '[88, 90, 92, 93, 94]'),
        ('歐美', 'Region', 90, '強調身體曲線與自信氣場，風格大膽直接，常運用簡約單品穿出高級感。', '[85, 87, 88, 89, 90]'),
        ('英倫', 'Region', 86, '經典的風衣、格紋與牛津鞋，展現紳士淑女般的學院與復古氣息。', '[80, 82, 84, 85, 86]'),
        ('法式', 'Region', 93, 'Effortless Chic，條紋衫、亂髮與紅唇，展現不經意的優雅與慵懶。', '[90, 91, 92, 93, 93]'),
        ('波西米亞', 'Region', 80, '流蘇、印花長裙與編織元素，展現自由奔放的流浪藝術家氣息。', '[70, 75, 78, 79, 80]'),
        ('通勤', 'Scene', 95, '專業且得體，西裝褲、襯衫與跟鞋的搭配，適合職場環境。', '[90, 92, 93, 94, 95]'),
        ('休閒', 'Scene', 98, 'T恤、牛仔褲與球鞋，強調舒適自在，適合週末與日常生活的輕鬆裝扮。', '[95, 96, 97, 97, 98]'),
        ('田園', 'Scene', 84, 'Cottagecore，碎花洋裝、草帽與編織包，展現回歸自然的清新感。', '[75, 80, 82, 83, 84]'),
        ('校園', 'Scene', 88, '百褶裙、針織背心與帆布鞋，洋溢著青春活力的學生氣息。', '[82, 85, 86, 87, 88]'),
        ('運動', 'Scene', 91, 'Athleisure，瑜珈褲、衛衣與機能材質，兼具健身與時尚的混搭風。', '[85, 88, 89, 90, 91]'),
        ('Party', 'Scene', 87, '亮片、絲絨與大膽剪裁，適合夜晚聚會與派對的吸睛裝扮。', '[80, 82, 85, 86, 87]'),
        ('約會', 'Scene', 93, '柔和色系、修身剪裁與浪漫元素，展現迷人魅力的心動穿搭。', '[88, 90, 91, 92, 93]'),
        ('度假', 'Scene', 89, '飄逸長裙、亞麻材質與鮮豔色彩，適合海島與旅行的放鬆風格。', '[82, 85, 87, 88, 89]'),
        ('新中式', 'Design', 96, '將盤扣、立領等傳統元素融入現代西裝或休閒剪裁，展現摩登東方韻味。',
         '[85, 88, 92, 95, 96]'),
        ('淑女', 'Design', 82, '端莊典雅，常運用蕾絲、珍珠與粉嫩色系，展現溫柔氣質。', '[78, 80, 81, 81, 82]'),
        ('名媛', 'Design', 88, '小香風毛呢、精緻套裝與高品質面料，展現富家千金的高貴感。', '[82, 85, 86, 87, 88]'),
        ('瑞麗', 'Design', 78, '日雜甜美風，強調層次感、蝴蝶結與細節裝飾，展現可愛女人味。', '[70, 75, 76, 77, 78]'),
        ('簡約', 'Design', 94, '去除多餘裝飾，強調版型與材質，耐看且百搭的基礎風格。', '[90, 91, 92, 93, 94]'),
        ('極簡', 'Design', 95, 'Minimalism，黑白灰中性色調，極致的線條與輪廓，展現冷靜高級感。', '[91, 92, 93, 94, 95]'),
        ('中性', 'Design', 86, 'Unisex，模糊性別界線，寬鬆剪裁與中性色調，展現帥氣隨性。', '[80, 82, 84, 85, 86]'),
        ('性冷淡', 'Design', 83, 'Normcore，低飽和度色系，寬鬆舒適，展現不食人間煙火的高冷感。', '[78, 80, 81, 82, 83]'),
        ('民族', 'Design', 76, '運用傳統印花、刺繡與手工藝元素，展現獨特文化底蘊。', '[70, 72, 74, 75, 76]'),
        ('戲劇', 'Design', 79, '誇張的輪廓、大墊肩或不規則剪裁，具備舞台張力的強烈風格。', '[72, 75, 77, 78, 79]'),
        ('復古', 'Design', 90, 'Vintage，汲取60-90年代的時尚元素，如高腰褲、波點與老花，展現懷舊風情。',
         '[85, 87, 88, 89, 90]'),
        ('Y2K', 'Trend', 92, '千禧辣妹風，低腰褲、金屬感、亮色與短版上衣，展現復古未來感。', '[85, 88, 90, 91, 92]'),
        ('嘻哈', 'Trend', 85, 'Oversize T恤、垮褲、金項鍊與球鞋，源自街頭文化的率性風格。', '[80, 82, 83, 84, 85]'),
        ('朋克', 'Trend', 78, 'Punk，皮革、鉚釘、格紋與破壞元素，展現反叛與不羈的個性。', '[72, 75, 76, 77, 78]'),
        ('嘻皮', 'Trend', 75, '紮染、喇叭褲與和平標誌，源自60年代的愛與和平運動風格。', '[70, 72, 73, 74, 75]'),
        ('甜酷', 'Trend', 91, '結合甜美少女元素與帥氣個性單品（如碎花裙配馬丁靴），展現反差魅力。', '[85, 88, 89, 90, 91]'),
        ('拜占庭', 'Art', 72, '金碧輝煌，運用鑲嵌珠寶、絲絨與重工刺繡，展現極致的奢華與莊嚴。', '[65, 68, 70, 71, 72]'),
        ('浪漫', 'Art', 88, '荷葉邊、蕾絲、薄紗與柔和色彩，充滿詩意與情感的唯美風格。', '[82, 85, 86, 87, 88]'),
        ('哥特', 'Art', 76, '黑色主調，蕾絲、馬甲與十字架元素，展現神秘、陰鬱且華麗的氣質。', '[70, 72, 74, 75, 76]'),
        ('巴洛克', 'Art', 74, '強調動態與裝飾，運用金線、錦緞與繁複圖騰，展現戲劇性的豪華感。', '[68, 70, 72, 73, 74]'),
        ('洛可可', 'Art', 75, '精緻細膩，粉嫩色系、蝴蝶結與蓬裙，展現輕盈、甜美與宮廷氣息。', '[68, 70, 73, 74, 75]'),
        ('洛麗塔', 'Art', 73, 'Lolita，層層疊疊的蕾絲裙、波奈特帽，追求洋娃娃般的精緻與夢幻。', '[68, 70, 71, 72, 73]'),
        ('維多利亞', 'Art', 77, '高領、羊腿袖、束腰與長裙，展現古典、保守且優雅的英式復古。', '[72, 74, 75, 76, 77]'),
        ('新古典', 'Art', 80, '簡潔典雅，強調垂墜感與對稱線條，展現如希臘女神般的高貴氣質。', '[75, 77, 78, 79, 80]'),
        ('超現實', 'Art', 70, '運用錯視圖案、奇異剪裁與夢境元素，挑戰常規視覺的藝術風格。', '[65, 67, 68, 69, 70]'),
        ('波普藝術', 'Art', 82, 'Pop Art，高飽和色彩、漫畫拼貼與重複圖案，展現活潑與通俗文化的趣味。',
         '[78, 80, 81, 81, 82]'),
        ('歐普', 'Art', 71, 'Op Art，運用黑白幾何與線條創造視覺錯視與律動感的迷幻風格。', '[65, 68, 69, 70, 71]'),
        ('未來色彩', 'Art', 84, '金屬光澤、霓虹色與科技感材質，展現對太空與未來的想像。', '[78, 80, 82, 83, 84]'),
        ('多巴胺', 'Art/Color', 97, '高飽和度的亮色系混搭，利用色彩心理學提振心情，傳遞快樂能量。',
         '[90, 93, 95, 96, 97]'),
        ('美拉德', 'Art/Color', 98, '秋冬必備的大地色系疊穿，以棕色、卡其、焦糖色為主，展現溫暖沈穩質感。',
         '[92, 95, 96, 97, 98]'),
        ('老錢風', 'Style', 95, 'Old Money，強調質感、中性色與低調奢華的經典風格，不顯露Logo。', '[90, 92, 93, 94, 95]')
    ]

    # [B] 網路即時資料
    try:
        web_trends = fetch_trends_from_web()
    except NameError:
        print("⚠️ fetch_trends_from_web 未定義，跳過網路搜尋")
        web_trends = []

    # [C] 合併與寫入
    all_trends = static_trends + web_trends
    print(f"📊 準備寫入 {len(all_trends)} 筆趨勢資料...")

    for kw, cat, score, desc, data in all_trends:
        c.execute(
            "INSERT OR REPLACE INTO trends (keyword, category, influence_score, description, data_points) VALUES (?, ?, ?, ?, ?)",
            (kw, cat, score, desc, data))

    # [D] 補充名人示範
    def add_celeb(trend_kw, name, desc):
        t = c.execute("SELECT id FROM trends WHERE keyword LIKE ?", (f'{trend_kw}%',)).fetchone()
        if t:
            c.execute("INSERT OR REPLACE INTO celebrity_looks (trend_id, celeb_name, description) VALUES (?, ?, ?)",
                      (t[0], name, desc))

    add_celeb('美拉德', 'Hailey Bieber', '經典的棕色長大衣搭配同色系針織。')
    add_celeb('多巴胺', '泫雅 HyunA', '色彩繽紛的撞色搭配。')
    add_celeb('新中式', '楊冪', '改良式旗袍與現代單品混搭。')
    add_celeb('Y2K', 'NewJeans', '青春活力的校園風格。')
    add_celeb('老錢風', 'Sofia Richie', '低調奢華的婚禮穿搭。')
    add_celeb('法式', 'Jeanne Damas', '經典的碎花洋裝與草編鞋。')
    add_celeb('極簡', 'Kendall Jenner', '俐落的黑白灰穿搭。')

    # [E] 初始化管理員與範例用戶
    pw = generate_password_hash('123456')
    c.execute("INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?, ?, ?, ?, 1)",
              ('admin@style.com', generate_password_hash('admin123'), '系統管理員', 'admin'))
    c.execute("INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?, ?, ?, ?, 1)",
              ('official@style.com', pw, 'Smart Style 官方', 'official'))
    c.execute("INSERT OR IGNORE INTO users (email, password, name, role, is_vip) VALUES (?, ?, ?, ?, 1)",
              ('expert@style.com', pw, 'Kevin 老師', 'expert'))

    # [F] 寫入商品
    default_img = 'https://placehold.co/400x300?text=Fashion+Item'
    items = [
        ('新中式刺繡盤扣上衣', 'top', 1280, default_img, '["新中式", "中國風", "復古", "約會"]'),
        ('美拉德燈芯絨寬褲', 'bottom', 990, default_img, '["美拉德", "復古", "通勤", "秋冬"]'),
        ('Y2K 千禧辣妹短T', 'top', 590, default_img, '["Y2K", "甜酷", "Party", "多巴胺"]'),
        ('日系亞麻襯衫', 'top', 990, default_img, '["日系", "簡約", "休閒"]'),
        ('高腰修身寬褲', 'bottom', 1490, default_img, '["歐美", "通勤", "顯瘦"]')
    ]
    for title, cat, price, img, tags in items:
        exist = c.execute("SELECT id FROM clothing_items WHERE title=?", (title,)).fetchone()
        if not exist:
            c.execute(
                "INSERT INTO clothing_items (title, category, brand, price, is_ad, image_path, tags) VALUES (?,?,?,?,?,?,?)",
                (title, cat, 'StyleSelect', price, 0, img, tags))

    conn.commit()
    conn.close()
    print("✅ 資料庫初始化完成！")

# --- 輔助函式 (Helper Functions) ---

def vip_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: return redirect(url_for('login_page'))
        if not session.get('is_vip'):
            flash('此功能限 VIP 使用', 'warning')
            return redirect(url_for('premium_landing'))
        return f(*args, **kwargs)

    return decorated_function


def get_weather_data(location="Taoyuan"):
    return {'temp': 29, 'condition': 'humid', 'humidity': 85, 'desc': '悶熱多雲'}


def check_style_fatigue(user_id):
    conn = get_db_connection()
    recent = conn.execute(
        'SELECT final_recommendation FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 3',
        (user_id,)).fetchall()
    conn.close()
    if len(recent) < 3: return False
    count = 0
    for r in recent:
        if r['final_recommendation'] and "簡約" in r['final_recommendation']:
            count += 1
    return count >= 3


def get_story_tag(life_stage):
    stories = {
        'student': ['圖書館的午後邂逅', '期末報告的自信戰袍', '社團成發的閃亮時刻'],
        'new_grad': ['第一次面試的沈穩', '週五下班的小酌時光', '給同事的好印象穿搭'],
        'career_change': ['重新定義自己的勇氣', '跨領域的專業展現', '跳脫舒適圈的嘗試'],
        'stable': ['週末家庭日的愜意', '主管會議的氣場', '一個人的質感旅行'],
        'explore': ['沒有目的地的漫遊', '嘗試沒穿過的色彩', '尋找第二個自己']
    }
    return random.choice(stories.get(life_stage, ['日常的好心情']))


def get_user_dislikes(user_data):
    try:
        prefs = json.loads(user_data.get('style_preferences', '{}'))
        return prefs.get('dislike', [])
    except:
        return []


def update_user_dislikes(user_id, tags):
    """更新用戶的避雷清單 (新增不喜歡的標籤)"""
    conn = get_db_connection()
    user = conn.execute("SELECT style_preferences FROM users WHERE id=?", (user_id,)).fetchone()
    if user:
        try:
            prefs = json.loads(user['style_preferences'])
        except:
            prefs = {'like': [], 'dislike': []}

        current_dislikes = set(prefs.get('dislike', []))
        for t in tags:
            current_dislikes.add(t)

        prefs['dislike'] = list(current_dislikes)
        conn.execute("UPDATE users SET style_preferences=? WHERE id=?", (json.dumps(prefs), user_id))
        conn.commit()
    conn.close()


def is_safe_recommendation(text, dislikes):
    for bad_tag in dislikes:
        if bad_tag in text:
            return False
    return True


def check_analysis_frequency(user_id):
    conn = get_db_connection()
    last_record = conn.execute(
        'SELECT created_at FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
        (user_id,)).fetchone()
    conn.close()
    if last_record:
        try:
            last_time = datetime.datetime.strptime(last_record['created_at'], '%Y-%m-%d %H:%M:%S')
            if (datetime.datetime.now() - last_time).total_seconds() < 300:
                return False
        except:
            pass
    return True


def verify_face_identity(user_id, current_img_path):
    # 模擬人臉驗證邏輯
    if 'stranger' in current_img_path:
        return False, "檢測到臉部特徵與您本人不符，請勿上傳他人照片。"
    return True, ""


# --- [新增] 矩陣座標計算函式 ---
def calculate_style_coordinates(face_data, body_data):
    """
    計算用戶在「量感 (Volume)」與「曲直 (Line)」軸上的位置
    分數範圍：0 (小量感/直線) ~ 10 (大量感/曲線)
    """
    volume_score = 5.0  # 預設中等
    curve_score = 5.0  # 預設平衡

    # 1. 從臉部分析 (Face)
    # 我們從分析紀錄中提取數據，若沒有則使用預設值
    if face_data and 'ratios' in face_data:
        r = face_data['ratios']
        # 臉長寬比 (Face Ratio L/W): 越長 -> 量感越大 (成熟)
        len_w = r.get('face_ratio (L/W)', 1.3)
        if len_w > 1.4: volume_score += 2
        if len_w < 1.2: volume_score -= 2

        # 下顎顴骨比 (Jaw/Cheek Ratio): 越寬/方 -> 越直 (硬朗)
        jaw_cheek = r.get('jaw_cheek_ratio', 0.8)
        if jaw_cheek > 0.85: curve_score -= 3  # 方臉偏直
        if jaw_cheek < 0.75: curve_score += 3  # 尖臉偏曲

    # 2. 從身形分析 (Body): 骨架大小修正量感
    if body_data and 'shape' in body_data:
        s = body_data['shape']
        if '倒三角' in s: curve_score -= 1  # 肩寬 -> 直線感
        if '梨型' in s: curve_score += 1  # 臀寬 -> 曲線感

    # 限制範圍 0-10
    return max(0, min(10, volume_score)), max(0, min(10, curve_score))


def get_quadrant_info(volume, curve, gender='female'):
    """根據座標回傳對應的象限資訊"""

    # 中心點判定 (4-6之間視為平衡)
    if 4 <= volume <= 6 and 4 <= curve <= 6:
        q_key = "CENTER"
    elif volume < 5:
        # 小量感 (年輕)
        q_key = "Q1" if curve >= 5 else "Q2"  # Q1:曲(柔), Q2:直(剛)
    else:
        # 大量感 (成熟)
        q_key = "Q3" if curve >= 5 else "Q4"  # Q3:曲(柔), Q4:直(剛)

    info = STYLE_MATRIX[q_key]
    # 根據性別選擇稱呼
    archetype = info['archetype'].get(gender, info['archetype']['female'])

    return {
        'code': q_key,
        'name': info['name'],
        'keywords': info['keywords'],
        'archetype': archetype,
        'clothing_guide': info['clothing_guide']
    }


# --- [更新版] 極限演化推論引擎 (V4.0 Matrix Fusion) ---
def analyze_style_logic(user_data, weather, event, variant, mode='normal'):
    """
    整合：模式攔截 -> 矩陣定位 -> 象限風格 -> 場合/天氣微調 -> 反向過濾
    """
    trace = []

    # ==========================================
    # 🛑 1. 模式攔截器 (Mode Interceptor) - 優先權最高
    # ==========================================

    if mode == 'low_energy':
        trace.append("MODE: Low Energy")
        if weather['temp'] >= 26:
            rec = "建議選擇「舒適休閒系」的短袖棉麻套裝，放鬆身心。"
        else:
            rec = "建議穿著柔軟的刷毛大學T (衛衣) 與棉褲，以舒適為主。"
        return rec, "MODE: Low Energy", "休息與充電"

    # 緊急場合優先處理
    if event:
        is_funeral = any(k in event for k in ['喪禮', '告別式', '公祭'])
        is_interview = any(k in event for k in ['面試', '求職', '重要會議'])

        if is_funeral:
            return ("【喪禮穿搭規範】請著全身黑色素面服裝，避免飾品，保持莊重。", "MODE: Funeral", "致意")
        if is_interview:
            return ("【面試穿搭規範】建議選擇「經典職場系」的深色正裝，展現專業與權威感。", "MODE: Interview", "專業")

    # ==========================================
    # 🧬 2. 矩陣定位 (Core Identity) - 找出底層風格
    # ==========================================

    # 嘗試解析用戶的臉型與身形數據 (從 user_data 中撈取，若無則用預設)
    try:
        f_data = json.loads(user_data.get('face_data', '{}'))
    except:
        f_data = {}
    try:
        b_data = json.loads(user_data.get('body_data', '{}'))
    except:
        b_data = {}

    # 計算座標
    vol, cur = calculate_style_coordinates(f_data, b_data)
    gender = user_data.get('gender', 'female')

    # 取得象限風格資訊
    style_info = get_quadrant_info(vol, cur, gender)

    trace.append(f"Matrix Pos: Vol={vol:.1f}, Line={cur:.1f} -> {style_info['code']}")
    trace.append(f"Archetype: {style_info['archetype']}")

    candidates = []
    suggested_tags = style_info['keywords'][:2]  # 預設標籤

    # A. 主風格建議
    candidates.append(f"經 AI 分析，您的風格定位為【{style_info['name']}】，視覺印象是{'、'.join(style_info['keywords'])}。")
    candidates.append(f"這對應到時尚原型中的「{style_info['archetype']}」。")
    candidates.append(f"💡 選衣建議：{style_info['clothing_guide']}")

    # ==========================================
    # 🌍 3. 環境與行程微調 (Context Layer)
    # ==========================================

    # [行程邏輯] 結合矩陣風格與行程需求
    if event and event != "無特別行程":
        trace.append(f"Context: {event}")
        if any(x in event for x in ['約會', '晚餐']):
            suggested_tags.append('浪漫')
            if style_info['code'] in ['Q2', 'Q4']:  # 如果原本是硬朗風格
                candidates.append(f"針對約會，建議在您的硬朗風格中加入一點柔軟材質（如絲巾或針織），增加親和力。")
            else:
                candidates.append("這正是發揮您風格優勢的時刻，盡情展現柔美魅力吧！")

        elif any(x in event for x in ['派對', 'Party']):
            suggested_tags.append('派對')
            candidates.append(
                f"派對場合，可以嘗試將您的風格元素（{style_info['clothing_guide'].split('、')[0]}）進行誇張化搭配。")

        elif any(x in event for x in ['上班', '工作']):
            suggested_tags.append('職場')
            if style_info['code'] in ['Q1', 'Q3']:  # 如果原本是柔和風格
                candidates.append("工作場合建議選擇剪裁較俐落的單品，平衡原本的柔和感，增加專業度。")

    # [天氣邏輯]
    if weather['temp'] >= 28:
        candidates.append("因應天氣炎熱，建議選擇該風格中的透氣輕薄款式，或適度露膚。")
    elif weather['temp'] <= 20:
        candidates.append("氣溫轉涼，建議利用「多層次疊穿」來增加造型的層次感，例如運用美拉德色系。")

    # [人生階段] (作為輔助建議)
    stage = user_data.get('life_stage', 'student')
    if stage == 'student':
        candidates.append("考量學生身份，建議尋找同風格的高CP值單品。")
    elif stage == 'new_grad':
        candidates.append("建議建立膠囊衣櫥，投資幾件符合您風格的高質感單品。")

    # ==========================================
    # 🔍 4. 最終過濾 (Final Filtering)
    # ==========================================

    dislikes = get_user_dislikes(user_data)
    final_advice = []

    for cand in candidates:
        if is_safe_recommendation(cand, dislikes):
            final_advice.append(cand)
        else:
            trace.append(f"FILTERED [Dislike]: Removed '{cand}'")

    # 安全備案
    if not final_advice:
        final_advice.append("建議選擇簡約的素色款式，這永遠是最安全的選擇。")

    # 生成 Hashtags
    main_tag = style_info['name'].split('/')[0].strip()
    all_tags = [main_tag] + suggested_tags
    unique_tags = list(set(all_tags))

    final_text = " ".join(final_advice) + f"\n\n🏷️ 推薦關鍵字：{' #'.join(unique_tags)}"

    # 故事模式標題
    story = f"今天，展現{style_info['keywords'][0]}的自己"

    return final_text, " -> ".join(trace), story, style_info


# ==========================================
#  頁面路由
# ==========================================

@app.route('/')
def index(): return render_template('index.html')


@app.route('/set_locale/<locale>')
def set_locale(locale):
    session['locale'] = locale
    if 'user_id' in session:
        conn = get_db_connection()
        conn.execute('UPDATE users SET locale = ? WHERE id = ?', (locale, session['user_id']))
        conn.commit()
        conn.close()
    return redirect(request.referrer or url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            if user['status'] == 'banned':
                flash('帳號已被停權', 'error')
                return render_template('login.html')

            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['role'] = user['role']
            session['is_vip'] = bool(user['is_vip'])
            session['is_admin'] = (user['role'] == 'admin')
            session['locale'] = user['locale'] if user['locale'] else 'zh_TW'

            if session['is_admin']:
                flash('歡迎回來，管理員', 'success')
                return redirect(url_for('admin_dashboard'))

            return redirect(url_for('shop_page'))
        flash('帳號或密碼錯誤', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        if not request.form.get('agree_tos'):
            flash('請先同意服務條款', 'error')
            return render_template('register.html', taxonomy=STYLE_TAXONOMY)

        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        hashed_pw = generate_password_hash(password)

        issues = request.form.getlist('issues')
        style_likes = request.form.getlist('style_like')
        style_dislikes = request.form.getlist('style_dislike')
        style_pref = {'like': style_likes, 'dislike': style_dislikes}
        colors = request.form.getlist('colors')
        gender = request.form.get('gender')

        try:
            age = int(request.form.get('age')) if request.form.get('age') else None
            height = float(request.form.get('height')) if request.form.get('height') else None
            weight = float(request.form.get('weight')) if request.form.get('weight') else None
        except ValueError:
            age, height, weight = None, None, None

        try:
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO users (
                    email, password, name, tos_version,
                    gender, age, height, weight,
                    clothing_issues, style_preferences, color_preferences,
                    life_stage, culture_pref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email, hashed_pw, name, '1.0',
                gender, age, height, weight,
                json.dumps(issues, ensure_ascii=False),
                json.dumps(style_pref, ensure_ascii=False),
                json.dumps(colors, ensure_ascii=False),
                request.form.get('life_stage', 'student'),
                request.form.get('culture', 5)
            ))
            conn.commit()
            conn.close()

            conn = get_db_connection()
            new_user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            conn.close()

            session['user_id'] = new_user['id']
            session['user_name'] = new_user['name']
            session['role'] = new_user['role']
            session['is_vip'] = bool(new_user['is_vip'])
            session['locale'] = 'zh_TW'

            flash('註冊成功！AI 已根據您的偏好為您準備好專屬推薦。', 'success')
            return redirect(url_for('shop_page'))

        except sqlite3.IntegrityError:
            flash('Email 已存在，請直接登入', 'error')
            return redirect(url_for('login_page'))

    return render_template('register.html', taxonomy=STYLE_TAXONOMY)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/daily_guide')
def daily_guide_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    weather = get_weather_data()
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = get_db_connection()
    events = conn.execute('SELECT * FROM calendar_events WHERE user_id = ? AND date_str = ?',
                          (session['user_id'], today)).fetchall()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    event_title = events[0]['title'] if events else "無特別行程"
    advice_text, _, story = analyze_style_logic(dict(user), weather, event_title, 'A')
    return render_template('daily_guide.html', weather=weather, events=events, today=today, advice=[advice_text],
                           story=story)


@app.route('/shop')
def shop_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    locale = session.get('locale', 'zh_TW')
    try:
        currency = TRANSLATIONS[locale]['currency']
    except:
        currency = 'NT$' if locale == 'zh_TW' else 'US$'

    category = request.args.get('category')
    try:
        min_price = int(request.args.get('min', 0))
        max_price = int(request.args.get('max', 10000))
    except:
        min_price = 0
        max_price = 10000

    conn = get_db_connection()
    try:
        conn.execute(
            'UPDATE analysis_history SET is_converted = 1 WHERE user_id = ? AND id = (SELECT MAX(id) FROM analysis_history WHERE user_id = ?)',
            (session['user_id'], session['user_id']))
        conn.commit()
    except Exception as e:
        print(f"Tracking error: {e}")

    sql = "SELECT * FROM clothing_items WHERE price BETWEEN ? AND ?"
    params = [min_price, max_price]
    if category:
        sql += " AND category = ?"
        params.append(category)

    local_rows = conn.execute(sql + " ORDER BY is_ad DESC, created_at DESC", params).fetchall()
    conn.close()
    display_items = []

    for i in local_rows:
        price = i['price']
        if locale == 'en_US': price = int(price / 30)
        display_items.append({
            'id': i['id'], 'title': i['title'], 'brand': i['brand'], 'price': price,
            'image': i['image_path'], 'is_ad': i['is_ad'], 'trust': i['trust_score'],
            'is_external': False, 'link': None
        })

    try:
        external_raw = shop_adapter.fetch_items(category, min_price, max_price)
        for i in external_raw:
            price = i['price']
            if locale == 'en_US': price = int(price / 30)
            display_items.append({
                'id': i['id'], 'title': i['title'], 'brand': i['brand'], 'price': price,
                'image': i['image'],  # [Fixed] External Adapter now returns 'image' key
                'is_ad': False, 'trust': i['trust_score'],
                'is_external': True, 'link': i['link']
            })
    except Exception as e:
        print(f"External fetch error: {e}")

    random.shuffle(display_items)
    display_items.sort(key=lambda x: x['is_ad'], reverse=True)

    return render_template('shop.html', items=display_items, currency=currency, locale=locale,
                           current_cat=category, min_p=min_price, max_p=max_price)


@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    if request.method == 'POST':
        conn.execute('''UPDATE users SET name=?, life_stage=?, maturity_level=?, culture_pref=? WHERE id=?''',
                     (request.form.get('name'), request.form.get('life_stage'), request.form.get('maturity'),
                      request.form.get('culture'), session['user_id']))
        conn.commit()
        session['user_name'] = request.form.get('name')
        flash('個人檔案與人生階段已更新', 'success')
        return redirect(url_for('profile_page'))
    user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    conn.close()
    p_data = {
        'email': user['email'], 'name': user['name'], 'is_vip': user['is_vip'], 'role': user['role'],
        'life_stage': user['life_stage'], 'maturity': user['maturity_level'], 'culture': user['culture_pref'],
        'issues': json.loads(user['clothing_issues']) if user['clothing_issues'] else [],
        'styles': json.loads(user['style_preferences']) if user['style_preferences'] else {'like': [], 'dislike': []},
    }
    return render_template('profile.html', p=p_data)


@app.route('/settings')
def settings_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template('settings.html', user=user)


@app.route('/analysis')
def analysis_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    return render_template('analysis.html')


@app.route('/try_on')
def try_on_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    return render_template('try_on.html')


@app.route('/history')
def history_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    analyses = conn.execute('SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC',
                            (session['user_id'],)).fetchall()
    conn.close()
    parsed = []
    for r in analyses:
        try:
            f = json.loads(r['face_data']) if r['face_data'] else {}
            rec = json.loads(r['final_recommendation']) if r['final_recommendation'] else {}
            parsed.append(
                {'id': r['id'], 'date': r['created_at'], 'img': r['user_image_path'], 'face': f.get('shape', ''),
                 'style': rec.get('summary', '無'), 'trace': r['logic_trace']})
        except:
            pass
    return render_template('history.html', analyses=parsed)


@app.route('/lab')
def lab_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    tracking = conn.execute('SELECT * FROM body_tracking WHERE user_id = ? ORDER BY recorded_at ASC',
                            (session['user_id'],)).fetchall()
    last_analysis = conn.execute('SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
                                 (session['user_id'],)).fetchone()
    analysis_data = {}
    if last_analysis:
        try:
            analysis_data['face'] = json.loads(last_analysis['face_data'])
            analysis_data['body'] = json.loads(last_analysis['body_data'])
            analysis_data['rec'] = json.loads(last_analysis['final_recommendation'])
        except:
            pass
    conn.close()
    chart_labels = [(t['recorded_at'][:10] if t['recorded_at'] else 'Unknown') for t in tracking]
    chart_weights = [t['weight'] for t in tracking]
    return render_template('lab.html', tracking=tracking, labels=chart_labels, weights=chart_weights,
                           analysis=analysis_data)


@app.route('/community')
def community_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    posts = conn.execute('''
        SELECT p.*, u.name as user_name, u.role as user_role, u.is_vip, u.id as author_id,
        (SELECT COUNT(*) FROM likes WHERE post_id = p.id AND user_id = ?) as is_liked
        FROM posts p JOIN users u ON p.user_id = u.id ORDER BY p.created_at DESC
    ''', (session['user_id'],)).fetchall()
    posts_data = []
    for p in posts:
        comments = conn.execute(
            'SELECT c.*, u.name as commenter_name, u.role as commenter_role FROM comments c JOIN users u ON c.user_id = u.id WHERE c.post_id = ?',
            (p['id'],)).fetchall()
        total = p['poll_yes'] + p['poll_no']
        yes_pct = int((p['poll_yes'] / total) * 100) if total > 0 else 0
        posts_data.append({
            'id': p['id'], 'image': p['image_path'], 'content': p['content'],
            'author_name': "匿名" if p['is_anonymous'] else p['user_name'],
            'author_role': p['user_role'], 'is_vip': p['is_vip'],
            'author_id': p['author_id'], 'is_anonymous': p['is_anonymous'],
            'is_qa': p['is_qa'], 'poll_yes': p['poll_yes'], 'poll_no': p['poll_no'], 'yes_percent': yes_pct,
            'tags': json.loads(p['tags']) if p['tags'] else [],
            'likes_count': p['likes_count'], 'is_liked': p['is_liked'] > 0, 'comments': comments
        })
    conn.close()
    return render_template('community.html', posts=posts_data)


@app.route('/community/new', methods=['GET', 'POST'])
def new_post():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        unique = f"post_{uuid.uuid4()}_{filename}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique))
        safe_content = ContentSafety.sanitize(request.form.get('content'))
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO posts (user_id, image_path, content, tags, is_anonymous, is_qa) VALUES (?, ?, ?, ?, ?, ?)',
            (session['user_id'], f"uploads/{unique}", safe_content,
             json.dumps(request.form.get('tags').split(',')),
             request.form.get('is_anonymous') == 'on', request.form.get('is_qa') == 'on'))
        conn.commit()
        conn.close()
        return redirect(url_for('community_page'))
    return render_template('post_new.html')


@app.route('/premium')
def premium_landing():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    return render_template('premium.html')


@app.route('/premium/chat')
@vip_required
def chat_page():
    conn = get_db_connection()
    logs = conn.execute('SELECT * FROM chat_logs WHERE user_id = ? ORDER BY created_at ASC',
                        (session['user_id'],)).fetchall()
    conn.close()
    return render_template('chat_consultant.html', logs=logs)


@app.route('/premium/calendar')
@vip_required
def calendar_page():
    conn = get_db_connection()
    events = conn.execute('SELECT * FROM calendar_events WHERE user_id = ?', (session['user_id'],)).fetchall()
    conn.close()
    events_json = [{'title': e['title'], 'start': e['date_str'], 'description': e['outfit_desc']} for e in events]
    return render_template('calendar.html', events=json.dumps(events_json))


@app.route('/legal/terms')
def terms_page(): return render_template('legal.html', type='terms')


@app.route('/legal/wellness')
def wellness_page(): return render_template('legal.html', type='wellness')


@app.route('/search')
def search_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    return render_template('search.html')


# --- 後台分析輔助函式 ---
def calculate_fairness_metrics():
    conn = get_db_connection()
    query = '''
        SELECT 
            CASE WHEN body_data LIKE '%沙漏%' THEN 'Hourglass' 
                 WHEN body_data LIKE '%梨%' THEN 'Pear' 
                 ELSE 'Others' END as body_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_incorrect = 1 THEN 1 ELSE 0 END) as errors
        FROM analysis_history
        GROUP BY body_type
    '''
    rows = conn.execute(query).fetchall()
    conn.close()
    result = []
    for r in rows:
        rate = round((r['errors'] / r['total'] * 100), 1) if r['total'] > 0 else 0
        status = 'Normal' if rate < 15 else 'Bias Detected'
        result.append({'type': r['body_type'], 'rate': rate, 'status': status, 'count': r['total']})
    return result


def get_trend_analysis():
    return {
        'years': ['2023', '2024', '2025'],
        'styles': {'Minimalist': [30, 35, 40], 'Streetwear': [40, 30, 25], 'Vintage': [20, 25, 30]}
    }


@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin': return redirect(url_for('index'))
    conn = get_db_connection()
    stats = {
        'users': conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
        'posts': conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0],
        'reports': conn.execute('SELECT COUNT(*) FROM reports WHERE status="pending"').fetchone()[0],
    }
    bias_query = '''
        SELECT CASE WHEN body_data LIKE '%沙漏%' THEN 'Hourglass' ELSE 'Others' END as body_type, COUNT(*) as count
        FROM analysis_history GROUP BY body_type
    '''
    bias_data = conn.execute(bias_query).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM analysis_history').fetchone()[0]
    errors = conn.execute('SELECT COUNT(*) FROM analysis_history WHERE is_incorrect = 1').fetchone()[0]
    error_rate = round((errors / total * 100), 1) if total > 0 else 0

    def get_rate(v):
        t = conn.execute("SELECT COUNT(*) FROM analysis_history WHERE ab_variant = ?", (v,)).fetchone()[0]
        c = conn.execute("SELECT COUNT(*) FROM analysis_history WHERE ab_variant = ? AND is_converted = 1",
                         (v,)).fetchone()[0]
        return {'count': t, 'rate': round((c / t * 100), 1) if t > 0 else 0}

    ai_stats = {'total': total, 'error_rate': error_rate, 'ab_test': {'A': get_rate('A'), 'B': get_rate('B')}}

    try:
        fairness_data = calculate_fairness_metrics()
        trend_data = get_trend_analysis()
    except:
        fairness_data = []
        trend_data = {}

    feature_flags = [
        {'name': 'Beta: 3D 試穿', 'status': 'active', 'usage': 'Low', 'action': '考慮下架'},
        {'name': 'Legacy: 舊版問卷', 'status': 'deprecated', 'usage': 'None', 'action': '已封存'},
        {'name': 'Core: 臉型分析', 'status': 'active', 'usage': 'High', 'action': '核心功能'}
    ]
    try:
        trend_config_row = conn.execute("SELECT value FROM system_configs WHERE key='trend_weights'").fetchone()
        current_trends = json.loads(trend_config_row['value']) if trend_config_row else {}
    except:
        current_trends = {}
    try:
        proposals = conn.execute('''
            SELECT p.*, u.name as user_name FROM style_proposals p 
            JOIN users u ON p.user_id = u.id WHERE p.status = 'pending' ORDER BY p.created_at DESC
        ''').fetchall()
    except:
        proposals = []

    items = conn.execute('SELECT * FROM clothing_items ORDER BY created_at DESC').fetchall()
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    reports = conn.execute(
        'SELECT r.*, u.name as reporter_name, p.content as post_content FROM reports r JOIN users u ON r.reporter_id = u.id LEFT JOIN posts p ON r.post_id = p.id ORDER BY r.created_at DESC').fetchall()
    conn.close()
    return render_template('admin.html', stats=stats, ai_stats=ai_stats, items=items, users=users, reports=reports,
                           model_version=CURRENT_MODEL_VERSION, bias_data=bias_data, fairness_data=fairness_data,
                           trend_data=trend_data, feature_flags=feature_flags, proposals=proposals,
                           current_trends=current_trends)


@app.route('/api/generate_full_report', methods=['POST'])
def generate_full_report_api():
    try:
        data = request.json
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        variant = 'A' if random.random() > 0.5 else 'B'

        # [修正 1] 將前端傳來的臉部與身形數據，合併進 user 資料中
        # 這樣 analyze_style_logic 才能讀到當下的分析結果
        user_data = dict(user)
        user_data['face_data'] = json.dumps(data.get('face_data', {}))
        user_data['body_data'] = json.dumps(data.get('body_data', {}))

        # 呼叫推論引擎
        rec_text, trace_log, story, style_info = analyze_style_logic(user_data, get_weather_data(), "一般分析", variant)

        final_rec = {
            'summary': ContentSafety.sanitize("AI 深度演化報告"),
            'advice': rec_text,
            'story': story,
            'confidence': random.randint(85, 99),
            'disclaimer': ContentSafety.DISCLAIMER,
            'model_ver': CURRENT_MODEL_VERSION,
            'logic_trace': trace_log,  # <--- [修正 2] 補上這裡原本缺少的逗號
            'hairstyle': style_info.get('hairstyle', '適合您的自然髮型'),
            'makeup': style_info.get('makeup', '適合您的妝容建議'),
            'accessories': style_info.get('accessories', '適合您的配件單品'),
            'archetype': style_info.get('name', '專屬風格')
        }

        # 寫入資料庫
        conn.execute('''INSERT INTO analysis_history (user_id, user_image_path, face_data, body_data, final_recommendation, ai_confidence, ab_variant, model_version, logic_trace, is_converted) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (session['user_id'], session.get('current_image_path', ''),
                      json.dumps(data.get('face_data')),
                      json.dumps(data.get('body_data')),
                      json.dumps(final_rec), final_rec['confidence'], variant,
                      CURRENT_MODEL_VERSION, trace_log, 0))
        conn.commit()
        conn.close()

        return jsonify({'status': 'success', 'result': final_rec})

    except Exception as e:
        print(f"Generate Report Error: {e}")
        return jsonify({'status': 'error', 'msg': '生成報告時發生錯誤，請稍後再試'}), 500

@app.route('/api/external/v1/analyze', methods=['POST'])
def external_api_analyze():
    api_key = request.headers.get('X-API-KEY')
    if api_key != API_ACCESS_KEY:
        return jsonify({'error': 'Unauthorized', 'message': 'Invalid API Key'}), 401
    rec = {
        'status': 'success',
        'model': CURRENT_MODEL_VERSION,
        'recommendation': '建議搭配高腰寬褲以修飾身形。',
        'confidence': 92,
        'trace_id': str(uuid.uuid4())
    }
    return jsonify(rec)


@app.route('/api/wear_feedback', methods=['POST'])
def wear_feedback_api():
    d = request.json
    conn = get_db_connection()
    conn.execute('INSERT INTO wear_logs (user_id, date_str, outfit_desc, feeling, rating) VALUES (?, ?, ?, ?, ?)',
                 (session['user_id'], datetime.datetime.now().strftime('%Y-%m-%d'), d['desc'], d['feeling'],
                  d['rating']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': 'AI 已學習您的穿著感受'})


@app.route('/api/dislike_item', methods=['POST'])
def dislike_item_api():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    item_id = request.json.get('item_id')
    conn = get_db_connection()
    item = conn.execute('SELECT tags, category FROM clothing_items WHERE id = ?', (item_id,)).fetchone()
    conn.close()
    if not item: return jsonify({'status': 'error', 'msg': '找不到商品'})
    try:
        tags = json.loads(item['tags'])
    except:
        tags = item['tags'].split(',') if item['tags'] else []
    if item['category']: tags.append(item['category'])

    update_user_dislikes(session['user_id'], tags)
    return jsonify({'status': 'success', 'msg': '系統已依據您的回饋進行調整。', 'learned_tags': tags})


@app.route('/api/mirror_mode', methods=['POST'])
def mirror_mode_api():
    problem = request.json.get('problem', '')
    solution = ""
    if '腿短' in problem or '比例' in problem:
        solution = "試著把上衣紮進去，或者換一雙與褲子同色的鞋子來延伸視覺。"
    elif '沒精神' in problem or '暗沈' in problem:
        solution = "塗個口紅，或是戴上一副亮金屬色的耳環，能立刻提亮臉部。"
    elif '胖' in problem or '臃腫' in problem:
        solution = "露出身上最細的部位（手腕、腳踝），或是加上一條腰帶。"
    else:
        solution = "您看起來很棒！自信就是最好的穿搭。試著挺胸看看？"
    return jsonify({'status': 'success', 'solution': solution})


@app.route('/api/comment_post/<int:id>', methods=['POST'])
def comment_post(id):
    content = request.form['content']
    if ContentSafety.check_mental_health(content):
        flash('我們注意到您似乎心情低落。請記得，您並不孤單，需要時請尋求專業協助。', 'warning')
        return redirect(url_for('wellness_page'))
    safe_content = ContentSafety.sanitize(content)
    conn = get_db_connection()
    conn.execute('INSERT INTO comments (user_id, post_id, content) VALUES (?, ?, ?)',
                 (session['user_id'], id, safe_content))
    conn.commit()
    conn.close()
    return redirect(url_for('community_page'))


@app.route('/api/chat_response', methods=['POST'])
def chat_response_api():
    msg = request.json.get('message', '')
    if ContentSafety.check_mental_health(msg):
        reply = "我很擔心您的狀況。雖然我是 AI，但我建議您尋求真人朋友或專業心理諮商師的協助。台灣衛福部安心專線：1925。"
    else:
        reply = ContentSafety.sanitize("建議嘗試更圓潤的線條來修飾身形。")
    conn = get_db_connection()
    conn.execute('INSERT INTO chat_logs (user_id, sender, message) VALUES (?, ?, ?)', (session['user_id'], 'user', msg))
    conn.execute('INSERT INTO chat_logs (user_id, sender, message) VALUES (?, ?, ?)', (session['user_id'], 'ai', reply))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'reply': reply})


@app.route('/api/analyze_face', methods=['POST'])
def analyze_face_api():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    if not check_analysis_frequency(session['user_id']):
        return jsonify({'status': 'warning', 'msg': '請稍作休息，5分鐘後再試。', 'data': None})
    try:
        file = request.files['image']
        unique = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(save_path)
        session['current_image_path'] = f"uploads/{unique}"
    except Exception as e:
        return jsonify({'status': 'error', 'msg': '圖片上傳失敗'}), 400

    is_valid, err_msg = verify_face_identity(session['user_id'], save_path)
    if not is_valid:
        os.remove(save_path)
        return jsonify({'status': 'error', 'msg': err_msg})

    try:
        ai_result, error_msg = face_engine.analyze(save_path)
        if error_msg:
            try:
                os.remove(save_path)
            except:
                pass
            return jsonify({'status': 'error', 'msg': error_msg})
        return jsonify(
            {'status': 'success', 'msg': '分析完成', 'disclaimer': ContentSafety.DISCLAIMER, 'data': ai_result})
    except NameError:
        return jsonify({'status': 'error', 'msg': '系統錯誤：AI 引擎尚未初始化'}), 500
    except Exception as e:
        print(f"Analysis Error: {e}")
        return jsonify({'status': 'error', 'msg': '影像處理發生未知錯誤'}), 500


@app.route('/api/analyze_body', methods=['POST'])
def analyze_body_api():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    current_img_rel = session.get('current_image_path')
    analyze_target = None

    if 'image' in request.files:
        # 直接上傳新照片
        file = request.files['image']
        unique = f"body_{uuid.uuid4()}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(save_path)
        analyze_target = save_path
    elif current_img_rel:
        # 使用 Session 中的舊照片，需正確還原路徑
        possible_paths = [
            os.path.join(app.root_path, 'static', current_img_rel),
            os.path.join(app.root_path, current_img_rel)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                analyze_target = p
                break

        if not analyze_target:
            return jsonify({'status': 'error', 'msg': '找不到照片，請重新上傳'}), 400
    else:
        return jsonify({'status': 'error', 'msg': '請上傳全身照片'}), 400

    try:
        ai_result, error_msg = body_engine.analyze(analyze_target)

        if error_msg:
            return jsonify({'status': 'error', 'msg': error_msg})

        return jsonify({
            'status': 'success',
            'data': ai_result
        })

    except NameError:
        return jsonify({'status': 'error', 'msg': '系統錯誤：Body AI 引擎尚未初始化'}), 500
    except Exception as e:
        print(f"Body Analysis Error: {e}")
        return jsonify({'status': 'error', 'msg': '身形分析失敗，請確認照片包含完整身體'}), 500

@app.route('/api/like_post/<int:id>', methods=['POST'])
def like_post(id):
    conn = get_db_connection()
    exist = conn.execute('SELECT * FROM likes WHERE user_id=? AND post_id=?', (session['user_id'], id)).fetchone()
    if exist:
        conn.execute('DELETE FROM likes WHERE user_id=? AND post_id=?', (session['user_id'], id))
        conn.execute('UPDATE posts SET likes_count = likes_count - 1 WHERE id=?', (id,))
        act = 'unliked'
    else:
        conn.execute('INSERT INTO likes (user_id, post_id) VALUES (?, ?)', (session['user_id'], id))
        conn.execute('UPDATE posts SET likes_count = likes_count + 1 WHERE id=?', (id,))
        act = 'liked'
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'action': act})


@app.route('/api/vote_post', methods=['POST'])
def vote_post_api():
    pid, vote = request.json.get('post_id'), request.json.get('vote')
    conn = get_db_connection()
    if vote == 'yes':
        conn.execute('UPDATE posts SET poll_yes = poll_yes + 1 WHERE id = ?', (pid,))
    else:
        conn.execute('UPDATE posts SET poll_no = poll_no + 1 WHERE id = ?', (pid,))
    conn.commit()
    post = conn.execute('SELECT poll_yes, poll_no FROM posts WHERE id = ?', (pid,)).fetchone()
    conn.close()
    total = post['poll_yes'] + post['poll_no']
    pct = int((post['poll_yes'] / total) * 100) if total > 0 else 0
    return jsonify({'status': 'success', 'yes': post['poll_yes'], 'no': post['poll_no'], 'percent': pct})


@app.route('/api/ai_explain_vote', methods=['POST'])
def ai_explain_vote_api():
    post_id = request.json.get('post_id')
    yes_votes = request.json.get('yes', 0)
    no_votes = request.json.get('no', 0)
    conn = get_db_connection()
    post = conn.execute('SELECT image_path FROM posts WHERE id = ?', (post_id,)).fetchone()
    conn.close()
    if not post: return jsonify({'status': 'error', 'reason': '找不到原始貼文'})
    image_path = post['image_path']
    if image_path.startswith('uploads/'):
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
    else:
        full_path = os.path.join(app.root_path, image_path)
    if not os.path.exists(full_path): return jsonify({'status': 'error', 'reason': '圖片檔案遺失'})

    total = yes_votes + no_votes
    if total == 0: return jsonify({'status': 'success', 'reason': "尚無足夠投票數據進行分析。"})
    vote_trend = 'popular' if yes_votes >= no_votes else 'unpopular'
    try:
        explanation = vote_engine.analyze(full_path, vote_trend)
        final_reason = f"大家喜歡這套穿搭！{explanation}" if vote_trend == 'popular' else f"大家覺得還有進步空間。{explanation}"
        return jsonify({'status': 'success', 'reason': final_reason})
    except Exception as e:
        print(e)
        return jsonify({'status': 'success', 'reason': "AI 分析連線逾時，請稍後再試。"})


@app.route('/api/follow_user/<int:id>', methods=['POST'])
def follow_user(id):
    conn = get_db_connection()
    exist = conn.execute('SELECT * FROM follows WHERE follower_id=? AND followed_id=?',
                         (session['user_id'], id)).fetchone()
    if exist:
        conn.execute('DELETE FROM follows WHERE follower_id=? AND followed_id=?', (session['user_id'], id))
        act = 'unfollowed'
    else:
        conn.execute('INSERT INTO follows (follower_id, followed_id) VALUES (?, ?)', (session['user_id'], id))
        act = 'followed'
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'action': act})


@app.route('/api/report_post/<int:id>', methods=['POST'])
def report_post(id):
    conn = get_db_connection()
    conn.execute('INSERT INTO reports (reporter_id, post_id, reason) VALUES (?, ?, ?)',
                 (session['user_id'], id, request.form.get('reason')))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/admin/add_item', methods=['POST'])
def admin_add_item():
    file = request.files['image']
    unique = f"off_{uuid.uuid4()}_{file.filename}"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique))
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO clothing_items (image_path, title, category, tags, brand, price, is_ad) VALUES (?,?,?,?,?,?,?)',
        (f"uploads/{unique}", request.form['title'], request.form['category'], request.form['tags'],
         request.form['brand'], request.form['price'], request.form.get('is_ad') == 'on'))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_item/<int:id>')
def admin_delete_item(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM clothing_items WHERE id=?', (id,))
    conn.commit()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/ban_user/<int:id>')
def admin_ban_user(id):
    conn = get_db_connection()
    conn.execute("UPDATE users SET status='banned' WHERE id=?", (id,))
    conn.commit()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/unban_user/<int:id>')
def admin_unban_user(id):
    conn = get_db_connection()
    conn.execute("UPDATE users SET status='active' WHERE id=?", (id,))
    conn.commit()
    return redirect(url_for('admin_dashboard'))


@app.route('/api/resolve_report/<int:id>')
def resolve_report(id):
    conn = get_db_connection()
    conn.execute("UPDATE reports SET status='resolved' WHERE id=?", (id,))
    conn.commit()
    return redirect(url_for('admin_dashboard'))


@app.route('/api/upgrade_vip', methods=['POST'])
def upgrade_vip():
    conn = get_db_connection()
    conn.execute('UPDATE users SET is_vip=1 WHERE id=?', (session['user_id'],))
    conn.commit()
    conn.close()
    session['is_vip'] = True
    return jsonify({'status': 'success'})


@app.route('/api/generate_pdf')
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Smart Style Report", ln=1, align='C')
    pdf.output("static/uploads/report.pdf")
    return send_file("static/uploads/report.pdf", as_attachment=True)


@app.route('/api/calendar/add', methods=['POST'])
def calendar_add_api():
    d = request.json
    conn = get_db_connection()
    conn.execute('INSERT INTO calendar_events (user_id, date_str, title, outfit_desc) VALUES (?, ?, ?, ?)',
                 (session['user_id'], d['date'], d['title'], d['desc']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/api/update_consent', methods=['POST'])
def update_consent():
    conn = get_db_connection()
    conn.execute('UPDATE users SET data_consent=? WHERE id=?', (request.json.get('consent'), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': '更新成功'})


@app.route('/api/update_privacy_settings', methods=['POST'])
def update_privacy_settings():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    data = request.json
    policy = data.get('photo_policy', '30_days')
    ai_consent = 1 if data.get('ai_consent') == True else 0
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET photo_policy = ?, ai_training_consent = ? WHERE id = ?',
                     (policy, ai_consent, session['user_id']))
        conn.commit()
        msg = '隱私設定已更新'
    except Exception as e:
        msg = '更新失敗，請確認資料庫已升級'
        print(e)
    conn.close()
    return jsonify({'status': 'success', 'msg': msg})


@app.route('/api/delete_all_photos', methods=['POST'])
def delete_all_photos():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    user_id = session['user_id']
    conn = get_db_connection()
    records = conn.execute('SELECT user_image_path FROM analysis_history WHERE user_id = ?', (user_id,)).fetchall()
    deleted_count = 0
    for r in records:
        path = r['user_image_path']
        if path and 'default' not in path:
            if path.startswith('static/'):
                full_path = os.path.join(app.root_path, path)
            else:
                full_path = os.path.join(app.root_path, 'static', path)
            try:
                if os.path.exists(full_path):
                    os.remove(full_path)
                    deleted_count += 1
            except Exception as e:
                print(f"刪除失敗: {e}")
    conn.execute('UPDATE analysis_history SET user_image_path = NULL WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': f'已銷毀 {deleted_count} 張歷史照片，您的隱私已受保護。'})


@app.route('/api/download_my_data')
def download_my_data():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    uid = session['user_id']
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    history = conn.execute('SELECT * FROM analysis_history WHERE user_id = ?', (uid,)).fetchall()
    calendar = conn.execute('SELECT * FROM calendar_events WHERE user_id = ?', (uid,)).fetchall()
    wear_logs = conn.execute('SELECT * FROM wear_logs WHERE user_id = ?', (uid,)).fetchall()
    conn.close()
    export_data = {
        'user_profile': dict(user) if user else {},
        'analysis_history': [dict(row) for row in history],
        'calendar_events': [dict(row) for row in calendar],
        'wear_feedback': [dict(row) for row in wear_logs],
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_note': 'Smart Style Data Export'
    }
    if 'password' in export_data['user_profile']: del export_data['user_profile']['password']
    filename = f"smart_style_takeout_{uid}_{int(time.time())}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4, default=str)
    return send_file(path, as_attachment=True, download_name=f"My_Style_Data.json")


@app.route('/api/delete_account', methods=['POST'])
def delete_account():
    uid = session['user_id']
    conn = get_db_connection()
    for t in ['analysis_history', 'posts', 'comments', 'likes', 'follows', 'try_on_history', 'favorites',
              'calendar_events', 'chat_logs', 'body_tracking', 'wear_logs']:
        conn.execute(f'DELETE FROM {t} WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM users WHERE id=?', (uid,))
    conn.commit()
    conn.close()
    session.clear()
    return jsonify({'status': 'success', 'msg': '帳號已刪除'})


@app.route('/api/submit_proposal', methods=['POST'])
def submit_proposal():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    tag = request.form.get('tag_name')
    desc = request.form.get('description')
    if not tag: return jsonify({'status': 'error', 'msg': '標籤名稱不能為空'})
    conn = get_db_connection()
    conn.execute('INSERT INTO style_proposals (user_id, tag_name, description) VALUES (?, ?, ?)',
                 (session['user_id'], tag, desc))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': '感謝您的提案！審核通過後將加入系統標籤庫。'})


@app.route('/api/admin/review_proposal', methods=['POST'])
def review_proposal():
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403
    p_id = request.json.get('id')
    action = request.json.get('action')
    conn = get_db_connection()
    status = 'approved' if action == 'approve' else 'rejected'
    conn.execute('UPDATE style_proposals SET status = ? WHERE id = ?', (status, p_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/api/admin/update_trends', methods=['POST'])
def update_trends():
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403
    new_weights = request.json.get('weights')
    conn = get_db_connection()
    conn.execute('INSERT OR REPLACE INTO system_configs (key, value) VALUES (?, ?)',
                 ('trend_weights', json.dumps(new_weights)))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': '當季流行趨勢權重已更新'})


@app.route('/api/lab/track', methods=['POST'])
def lab_track_api():
    d = request.json
    conn = get_db_connection()
    conn.execute('INSERT INTO body_tracking (user_id, weight, waist, hip, note) VALUES (?, ?, ?, ?, ?)',
                 (session['user_id'], d['weight'], d['waist'], d['hip'], d['note']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


@app.route('/api/lab/correct', methods=['POST'])
def lab_correct_api(): return jsonify({'status': 'success', 'msg': 'AI 已更新'})


@app.route('/api/lab/mood', methods=['POST'])
def lab_mood_api():
    rec = {'happy': {'tone': '自信', 'text': '穿亮色!', 'items': ['洋裝']},
           'sad': {'tone': '溫暖', 'text': '穿衛衣', 'items': ['衛衣']}}
    return jsonify({'status': 'success', 'data': rec.get(request.json.get('mood'), rec['happy'])})


@app.route('/api/convert_size', methods=['POST'])
def convert_size_api():
    res = request.json.get('size') + (' (Asian)' if session.get('locale') == 'zh_TW' else ' (US)')
    return jsonify({'status': 'success', 'result': res})


@app.route('/api/try_on', methods=['POST'])
def try_on_api():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'msg': '請先登入'}), 401

    # ❌ [問題點 1] 這裡直接拿相對路徑，例如 "uploads/user.jpg"
    # 但檔案其實在 "static/uploads/user.jpg"，所以 os.path.exists 會回傳 False
    user_img_path = session.get('current_image_path')

    if not user_img_path or not os.path.exists(user_img_path):
        return jsonify({'status': 'error', 'msg': '請先在「身形分析」頁面上傳您的全身照片'}), 400

    clothing_id = request.json.get('clothing_id')
    if not clothing_id:
        return jsonify({'status': 'error', 'msg': '未選擇試穿衣物'}), 400

    conn = get_db_connection()
    cloth = conn.execute('SELECT * FROM clothing_items WHERE id = ?', (clothing_id,)).fetchone()
    conn.close()

    if not cloth:
        return jsonify({'status': 'error', 'msg': '找不到該件衣物'}), 404

    # ❌ [問題點 2] 衣物圖片也是同樣的問題
    cloth_img_path = cloth['image_path']
    if not os.path.exists(cloth_img_path):
        return jsonify({'status': 'error', 'msg': '衣物圖片遺失'}), 404

    try:
        category = 'upper_body'
        if '褲' in cloth['title'] or '裙' in cloth['title']:
            category = 'lower_body'

        # ❌ [問題點 3] 因為路徑錯誤，AI 引擎會因為找不到檔案而崩潰
        generated_path, error = vton_engine.generate(user_img_path, cloth_img_path, category)

        if error:
            return jsonify({'status': 'error', 'msg': f'AI 生成失敗 (可能排隊過久): {error}'}), 500

        new_filename = f"tryon_{uuid.uuid4()}.jpg"
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        shutil.copy(generated_path, target_path)

        result_url = f"uploads/{new_filename}"
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO try_on_history (user_id, original_img, cloth_img, result_img) 
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], user_img_path, cloth_img_path, result_url))
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'result_url': url_for('static', filename=result_url),
            'msg': '試穿合成成功！'
        })

    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'msg': '系統發生未預期錯誤'}), 500

@app.route('/api/search', methods=['POST'])
def search_api():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    keyword = request.json.get('keyword', '').strip()
    filters = request.json.get('filters', {})
    results = {'items': [], 'posts': []}
    conn = get_db_connection()
    item_sql = "SELECT * FROM clothing_items WHERE 1=1"
    item_params = []
    if keyword:
        item_sql += " AND (title LIKE ? OR tags LIKE ? OR brand LIKE ?)"
        k_param = f"%{keyword}%"
        item_params.extend([k_param, k_param, k_param])
    if 'category' in filters and filters['category']:
        item_sql += " AND category = ?"
        item_params.append(filters['category'])
    if 'min_price' in filters:
        item_sql += " AND price >= ?"
        item_params.append(filters['min_price'])
    if 'max_price' in filters:
        item_sql += " AND price <= ?"
        item_params.append(filters['max_price'])
    item_rows = conn.execute(item_sql + " ORDER BY created_at DESC LIMIT 20", item_params).fetchall()
    for r in item_rows:
        results['items'].append({
            'type': 'item', 'id': r['id'], 'title': r['title'], 'image': r['image_path'],
            'price': r['price'], 'brand': r['brand'], 'tags': r['tags']
        })
    post_sql = "SELECT p.*, u.name as user_name FROM posts p JOIN users u ON p.user_id = u.id WHERE 1=1"
    post_params = []
    if keyword:
        post_sql += " AND (p.content LIKE ? OR p.tags LIKE ?)"
        k_param = f"%{keyword}%"
        post_params.extend([k_param, k_param])
    post_rows = conn.execute(post_sql + " ORDER BY p.likes_count DESC LIMIT 20", post_params).fetchall()
    for r in post_rows:
        author = "匿名" if r['is_anonymous'] else r['user_name']
        results['posts'].append({
            'type': 'post', 'id': r['id'],
            'content': r['content'][:50] + "..." if len(r['content']) > 50 else r['content'],
            'image': r['image_path'], 'author': author, 'likes': r['likes_count']
        })
    conn.close()
    return jsonify({'status': 'success', 'results': results, 'count': len(results['items']) + len(results['posts'])})


@app.route('/api/add_favorite', methods=['POST'])
def add_favorite_api(): return jsonify({'status': 'success'})


@app.route('/api/report_error', methods=['POST'])
def report_error():
    conn = get_db_connection()
    conn.execute('UPDATE analysis_history SET is_incorrect=1, user_feedback=? WHERE id=?',
                 (request.json.get('feedback'), request.json.get('history_id')))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'msg': '感謝回饋'})


@app.route('/api/user/correct_profile', methods=['POST'])
def correct_user_profile():
    if 'user_id' not in session: return jsonify({'status': 'error', 'msg': '請先登入'}), 401
    new_shape = request.json.get('manual_shape')
    target = request.json.get('target')
    if not new_shape or target not in ['body', 'face']: return jsonify({'status': 'error', 'msg': '參數錯誤'})
    conn = get_db_connection()
    last_record = conn.execute(
        'SELECT id, body_data, face_data FROM analysis_history WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
        (session['user_id'],)).fetchone()
    if last_record:
        try:
            col = 'body_data' if target == 'body' else 'face_data'
            data = json.loads(last_record[col])
            data['shape'] = new_shape
            data['is_manual_corrected'] = True
            conn.execute(f'UPDATE analysis_history SET {col}=? WHERE id=?', (json.dumps(data), last_record['id']))
            conn.commit()
            msg = f'已將您的{target}校正為：{new_shape}，未來的推薦將以此為準。'
        except Exception as e:
            msg = '校正失敗，資料格式錯誤'
            print(e)
    else:
        msg = '尚無分析紀錄可供校正'
    conn.close()
    return jsonify({'status': 'success', 'msg': msg})


@app.route('/setup_db_final')
def setup_db_final():
    conn = get_db_connection()
    try:
        try:
            conn.execute("ALTER TABLE users ADD COLUMN maturity_level TEXT DEFAULT 'balanced'")
        except:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN culture_pref INTEGER DEFAULT 5")
        except:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN life_stage TEXT DEFAULT 'student'")
        except:
            pass
        try:
            conn.execute("ALTER TABLE analysis_history ADD COLUMN model_version TEXT")
        except:
            pass
        try:
            conn.execute("ALTER TABLE analysis_history ADD COLUMN logic_trace TEXT")
        except:
            pass
        try:
            conn.execute("ALTER TABLE analysis_history ADD COLUMN is_converted BOOLEAN DEFAULT 0")
        except:
            pass
        conn.execute('''CREATE TABLE IF NOT EXISTS wear_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
            date_str TEXT, outfit_desc TEXT, feeling TEXT, rating INTEGER,
            ai_adjustment_note TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        return "資料庫升級成功！包含所有進階功能。<a href='/'>回首頁</a>"
    except Exception as e:
        return f"資料庫檢查報告: {e} <a href='/'>回首頁</a>"


@app.route('/setup_admin')
def setup_admin(): return "OK"


@app.route('/api/research/export_report')
def export_research_report():
    if session.get('role') != 'admin': return jsonify({'status': 'error'}), 403
    conn = get_db_connection()
    shape_stats = {}
    rows = conn.execute("SELECT body_data FROM analysis_history").fetchall()
    for r in rows:
        try:
            b = json.loads(r['body_data'])
            shape = b.get('shape', 'Unknown')
            shape_stats[shape] = shape_stats.get(shape, 0) + 1
        except:
            pass
    psy_stats = conn.execute(
        'SELECT feeling, AVG(rating) as avg_score, COUNT(*) as count FROM wear_logs GROUP BY feeling').fetchall()
    psy_data = [{'feeling': r['feeling'], 'avg_score': round(r['avg_score'], 1), 'sample_size': r['count']} for r in
                psy_stats]
    conn.close()
    report = {
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'title': 'Smart Style 匿名研究數據報告',
        'modules': {'body_diversity': shape_stats, 'psychological_impact': psy_data,
                    'note': '本數據已去識別化，僅供學術研究使用。'}
    }
    filename = f"research_export_{int(time.time())}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    return send_file(path, as_attachment=True)


class TrendEngine:
    @staticmethod
    def calculate_compatibility(user_data, trend_tags):
        score = 70
        body_shape = 'Unknown'
        if user_data.get('body_data'):
            try:
                body = json.loads(user_data['body_data'])
                body_shape = body.get('shape', 'Unknown')
            except:
                pass
        if 'Oversize' in trend_tags and body_shape == '梨形': score += 15
        if 'Skinny' in trend_tags and body_shape == '蘋果型': score -= 20
        if 'Green' in trend_tags and 'Yellow' in user_data.get('skin_tone', ''): score -= 10
        return min(100, max(0, score))

    @staticmethod
    def forecast_trend(history_data):
        if not history_data or len(history_data) < 2: return 50
        slope = history_data[-1] - history_data[-2]
        prediction = history_data[-1] + slope
        return min(100, max(0, prediction))


@app.route('/trends')
def trends_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    trends_db = conn.execute('SELECT * FROM trends ORDER BY influence_score DESC').fetchall()
    today_seed = datetime.datetime.now().strftime('%Y%m%d')
    trend_list = []
    for t in trends_db:
        history = json.loads(t['data_points'])
        prediction = TrendEngine.forecast_trend(history)
        match_score = TrendEngine.calculate_compatibility(dict(user), t['category'])
        celebs_rows = conn.execute('SELECT * FROM celebrity_looks WHERE trend_id = ? LIMIT 3', (t['id'],)).fetchall()
        celebs_data = []
        for c in celebs_rows:
            c_dict = dict(c)
            unique_seed = f"{today_seed}_{c['id']}"
            rng = random.Random(unique_seed)
            c_dict['likes_count'] = rng.randint(10000, 500000)
            celebs_data.append(c_dict)
        trend_list.append({
            'id': t['id'], 'keyword': t['keyword'], 'status': t['status'], 'score': t['influence_score'],
            'description': t['description'], 'history': history, 'prediction': prediction,
            'match_score': match_score, 'celebs': celebs_data
        })
    conn.close()
    region = session.get('locale', 'zh_TW')
    region_filter = "台灣/亞洲趨勢" if region == 'zh_TW' else "歐美/全球趨勢"
    return render_template('trends.html', trends=trend_list, region=region_filter, user=user)


@app.route('/api/trend/match_celeb', methods=['POST'])
def match_celeb_style():
    return jsonify({
        'status': 'success', 'similarity': random.randint(70, 95), 'celeb_name': 'Jennie',
        'common_items': ['短版上衣', '高腰褲'], 'msg': '您的穿搭結構與 Jennie 的「機場時尚」高度相似！'
    })


@app.route('/smart_mirror')
def smart_mirror():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    last_analysis = conn.execute('SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
                                 (session['user_id'],)).fetchone()
    conn.close()
    recommendation = "今天還沒分析喔！"
    if last_analysis:
        try:
            rec_json = json.loads(last_analysis['final_recommendation'])
            recommendation = rec_json.get('summary', '保持自信！')
        except:
            pass
    weather = get_weather_data()
    return render_template('mirror.html', user=user, weather=weather, rec=recommendation)


@app.route('/api/check_ar_capability', methods=['POST'])
def check_ar_capability():
    device_type = request.json.get('device', 'unknown')
    if 'Mobile' in device_type:
        return jsonify({'status': 'success', 'ar_ready': True, 'msg': '📱 您的裝置支援 AR 試穿'})
    else:
        return jsonify({'status': 'warning', 'ar_ready': False, 'msg': '💻 AR 功能建議使用手機體驗'})


@app.route('/api/update_accessibility', methods=['POST'])
def update_accessibility():
    if 'user_id' not in session: return jsonify({'status': 'error'}), 401
    prefs = request.json
    conn = get_db_connection()
    conn.execute('UPDATE users SET accessibility_prefs = ? WHERE id = ?', (json.dumps(prefs), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    init_db()  # <--- 務必加上這一行！
    app.run(debug=True, port=5000)