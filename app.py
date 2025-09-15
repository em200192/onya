import os, json, re, difflib, math, random, time, base64, uuid, hashlib, pathlib
import httpx
from typing import Any, Dict, List, Optional, Tuple
from elevenlabs.client import ElevenLabs

import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import VoiceSettings
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from streamlit_mic_recorder import mic_recorder
import io, tempfile

# ==================== Page Config ====================
st.set_page_config(
    page_title="Onyx — Real AI Order Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ENV ====================
load_dotenv()


def env(k: str, d: Optional[str] = None) -> str:
    v = os.getenv(k)
    return v if v is not None else (d if d is not None else "")
ELEVENLABS_API_KEY = env("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = env("ELEVENLABS_VOICE_ID", "aCChyB4P5WEomwRsOKRh")
PICOVOICE_API_KEY = env("PICOVOICE_API_KEY")

OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")

CURRENCY = env("CURRENCY", "YER")
LANG_CODE_AR = env("LANG_CODE_AR", "1")
LANG_CODE_EN = env("LANG_CODE_EN", "2")
DEFAULT_COUNTRY_PREFIX = env("DEFAULT_COUNTRY_PREFIX", "+20")

CATEGORIES_URL = env("CATEGORIES_URL")
ITEMS_URL = env("ITEMS_URL")
ORDER_URL = env("ORDER_URL")

APP_VERSION_CODE = env("APP_VERSION_CODE", "8101")
P_DVS_TYP = int(env("P_DVS_TYP", "4"))
P_BRN_NO = int(env("P_BRN_NO", "1"))
CREDIT_GROUP_TYP = int(env("CREDIT_GROUP_TYP", "9"))
BILL_DOC_TYPE = int(env("BILL_DOC_TYPE", "2"))
DEVIC_TYP = int(env("DEVIC_TYP", "4"))
APP_TYP = int(env("APP_TYP", "8707"))
CLC_TBL_SRVC_FLG = int(env("CLC_TBL_SRVC_FLG", "1"))
NOT_CLC_SRVC_TYP = int(env("NOT_CLC_SRVC_TYP", "1"))
PYMNT_FLG = int(env("PYMNT_FLG", "0"))
RES_UNRES_ALL = int(env("RES_UNRES_ALL", "1"))
FRM_MNU_APP_FLG = int(env("FRM_MNU_APP_FLG", "1"))

TAX_INCLUSIVE = env("TAX_INCLUSIVE", "true").lower() in ["1", "true", "yes"]
DEFAULT_VAT_PERCENT = float(env("DEFAULT_VAT_PERCENT", "5"))
ORDER_TIMEOUT = int(env("ORDER_TIMEOUT", "60"))
DEBUG_ORDERS = env("DEBUG_ORDERS", "false").lower() in ["1", "true", "yes"]
FLASH_TTL_SEC = int(env("FLASH_TTL_SEC", "12"))

# ===== Persistent storage for chat & cart =====
BASE_CACHE_DIR = pathlib.Path(os.getenv("ONYX_CACHE_DIR", ".onyx_cache")).expanduser()
BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# TTS ENV (voice-out)
def _tts_model() -> str: return env("TTS_MODEL", "gpt-4o-mini-tts")


def _tts_voice() -> str: return env("TTS_VOICE", "alloy")


# ==================== Basic Validation ====================
def validate_env():
    missing = [k for k in ["OPENAI_API_KEY", "CATEGORIES_URL", "ITEMS_URL", "ORDER_URL"] if not env(k)]
    if missing:
        st.error("Missing in .env: " + ", ".join(missing))
        raise RuntimeError("Missing in .env: " + ", ".join(missing))


validate_env()

# ==================== OpenAI Client ====================
_openai = None


def ai() -> OpenAI:
    global _openai
    if _openai is None:
        httpx_client = httpx.Client(timeout=60.0)
        _openai = OpenAI(http_client=httpx_client)
    return _openai


# ==================== Arabic Normalization ====================
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
AR_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
AR_DIACRITICS = re.compile(r'[\u064B-\u0652\u0640]')

# ==================== Background Listening (JARVIS Mode) ====================

# ==================== Background Listening (JARVIS Mode) ====================
def listen_in_background():
    """
    Listens for a voice command in the background, detects pauses, and returns the transcribed text.
    """
    if "recognizer" not in ss:
        ss["recognizer"] = sr.Recognizer()
        ss["microphone"] = sr.Microphone()
        # === UPDATE: Set recognizer to adjust sensitivity automatically ===
        ss["recognizer"].dynamic_energy_threshold = True
        ss["recognizer"].pause_threshold = 1.0 # One full second of silence marks the end of a phrase
        # =================================================================

    r = ss["recognizer"]
    mic = ss["microphone"]

    try:
        with mic as source:
            # Adjust for ambient noise each time before listening
            st.toast("Calibrating microphone...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("Listening for your command...")
            audio = r.listen(source, timeout=5, phrase_time_limit=15)

        st.toast("🧠 Recognizing...")
        text = r.recognize_google(audio, language="ar-EG" if ss.get("lang") == "ar" else "en-US")
        st.toast(f"Heard: '{text}'") # Keep this for debugging
        return text.strip()

    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand that. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Recognition service error: {e}")
        return None
# ============================ End of Listening Code ============================
# ==================== JARVIS MODE: Wake Word & Command Listening ====================
# ==================== JARVIS MODE: Wake Word & Command Listening ====================
def listen_for_command():
    """
    Listens for a user's command after the wake word is detected.
    """
    if "recognizer" not in ss:
        ss["recognizer"] = sr.Recognizer()
        ss["microphone"] = sr.Microphone()
        ss["recognizer"].dynamic_energy_threshold = True
        ss["recognizer"].pause_threshold = 1.0

    r = ss["recognizer"]
    mic = ss["microphone"]

    try:
        with mic as source:
            st.toast("Calibrating for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("✅ Listening for your command...")
            audio = r.listen(source, timeout=5, phrase_time_limit=15)

        st.toast("🧠 Recognizing...")
        text = r.recognize_google(audio, language="ar-EG" if ss.get("lang") == "ar" else "en-US")
        return text.strip()

    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError as e:
        st.error(f"Recognition service error: {e}")
        return None

def run_wake_word_loop():
    """
    Listens continuously for the wake word "Jarvis" and then calls listen_for_command.
    """
    if not PICOVOICE_API_KEY:
        st.error("Picovoice API Key is not set in .env. JARVIS mode is disabled.")
        ss["jarvis_active"] = False; st.rerun(); return

    porcupine, pa, audio_stream = None, None, None
    try:
        porcupine = pvporcupine.create(access_key=PICOVOICE_API_KEY, keywords=['jarvis'])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
            input=True, frames_per_buffer=porcupine.frame_length
        )
        st.toast("👂 Listening for 'Jarvis'...")

        while ss.get("jarvis_active", False):
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            if porcupine.process(pcm) > -1:
                st.toast("✔️ Wake word detected!")
                command = listen_for_command()
                if command:
                    ss["last_user_utterance"] = command
                break # Exit loop to process the command
    except Exception as e:
        st.error(f"Error in JARVIS mode: {e}"); ss["jarvis_active"] = False
    finally:
        if porcupine: porcupine.delete()
        if audio_stream: audio_stream.close()
        if pa: pa.terminate()
        st.rerun()
# ============================ End of JARVIS Code ============================
# ============================ End of Listening Code ============================
def normalize_arabic(s: str) -> str:
    if not s: return ""
    s = AR_DIACRITICS.sub("", s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
         .replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
         .replace("ﻻ", "لا"))
    return s


def canonicalize(s: str, lang: str) -> str:
    s = (s or "").strip()
    if lang == "ar": s = normalize_arabic(s)
    s = s.lower().translate(AR_DIGITS).translate(AR_PERSIAN_DIGITS)
    return s


# ==================== HTTP & Money Helpers ====================
def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json", "AppVersionCode": str(APP_VERSION_CODE)}


def _to_float(v: Any, d: float = 0.0) -> float:
    try:
        if v is None: return d
        if isinstance(v, (int, float)): return float(v)
        return float(str(v).strip().replace(",", ""))
    except:
        return d


def _r2(x: float) -> float:
    return float(f"{_to_float(x):.2f}")


def money(v: float) -> str:
    try:
        return f"{float(v):,.2f} {CURRENCY}"
    except:
        return f"{v} {CURRENCY}"


def split_price_by_vat(price: float, vat_percent: float) -> Tuple[float, float, float]:
    r = _to_float(vat_percent) / 100.0;
    p = _to_float(price)
    if r <= 0: return (p, p, 0.0)
    if TAX_INCLUSIVE:
        gross = p;
        net = gross / (1.0 + r);
        vat = gross - net;
        return (_r2(net), _r2(gross), _r2(vat))
    net = p;
    gross = net * (1.0 + r);
    vat = gross - net;
    return (_r2(net), _r2(gross), _r2(vat))


def cart_total(cart: List[Dict[str, Any]]) -> float:
    return _r2(sum(_to_float(it.get("price", 0)) * _to_float(it.get("qty", 0)) for it in cart))


def _text_tokens(t: str) -> List[str]:
    return re.findall(r"[A-Za-z\u0600-\u06FF0-9]+", (t or "").lower().translate(AR_DIGITS))


def _normalize_phone(raw: Optional[str]) -> Optional[str]:
    if not raw: return None
    s = raw.translate(AR_DIGITS)
    s = re.sub(r"[^\d+]", "", s)
    if s.startswith("00"): s = "+" + s[2:]
    if not s.startswith("+"):
        s = DEFAULT_COUNTRY_PREFIX + re.sub(r"\D", "", s)
    digits_only = re.sub(r"\D", "", s)
    if 7 <= len(digits_only) <= 15:
        return s
    return None


def _post_with_retries(url: str, json_payload: Dict[str, Any], timeout: int, tries: int = 3, backoff: float = 0.75):
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            return requests.post(url, headers=_headers(), json=json_payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            time.sleep(backoff * attempt * (0.5 + random.random()))
    if last_exc: raise last_exc


# ==================== STT (Transcription) ====================
def _transcribe_bytes(raw: bytes, filename: str = "voice.wav", lang_hint: Optional[str] = None) -> Optional[str]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tf:
            tf.write(raw)
            tmp_path = tf.name

        main_model = env("TRANSCRIBE_MODEL", "whisper-1")
        fb_model = env("TRANSCRIBE_MODEL_FALLBACK", "gpt-4o-mini-transcribe")

        def build_params(model_name: str):
            p = {"model": model_name, "response_format": "text"}
            lh = (lang_hint or "").lower()
            if lh.startswith("ar"):
                p["language"] = "ar"
                p["prompt"] = "أسماء أكلات ومشروبات للمطاعم: برجر، شاورما، فاهيتا، لحم، جبن، عائلي، كبير، وسط، صغير"
            elif lh.startswith("en"):
                p["language"] = "en"
            return p

        try:
            with open(tmp_path, "rb") as f:
                r = ai().audio.transcriptions.create(file=f, **build_params(main_model))
        except Exception:
            with open(tmp_path, "rb") as f:
                r = ai().audio.transcriptions.create(file=f, **build_params(fb_model))

        return getattr(r, "text", None) or str(r)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except:
                pass


# ==================== Data Fetch: Categories & Items ====================
@st.cache_data(ttl=300)
def get_categories(lang_code: str) -> List[Dict[str, Any]]:
    r = _post_with_retries(CATEGORIES_URL, {"Value": {"P_LANG_NO": str(lang_code)}}, timeout=40)
    r.raise_for_status()
    groups = ((r.json() or {}).get("Data") or {}).get("FoodGroups") or []
    out = []
    for g in groups:
        out.append({
            "id": g.get("FOOD_GRP_NO") or g.get("U_ID"),
            "name_ar": (g.get("FOOD_GRP_L_NM") or g.get("FOOD_GRP_NM") or "").strip(),
            "name_en": (g.get("FOOD_GRP_F_NM") or "").strip() or (
                    g.get("FOOD_GRP_L_NM") or g.get("FOOD_GRP_NM") or "").strip(),
        })
    return [c for c in out if c["name_ar"] or c["name_en"]]


@st.cache_data(ttl=300)
def get_items_for_group(group_code: str, lang_code: str) -> List[Dict[str, Any]]:
    r = _post_with_retries(ITEMS_URL, {"Value": {"P_LANG_NO": str(lang_code), "P_FOOD_GRP": str(group_code)}},
                           timeout=60)
    r.raise_for_status()
    all_rows = ((r.json() or {}).get("Data") or {}).get("FoodItems") or []

    # === FIX: Manually filter items because the API does not apply the filter ===
    rows = [
        item for item in all_rows
        if str(item.get("FOOD_GRP_NO", "")).lstrip('0') == str(group_code).lstrip('0')
    ]
    # ============================ END OF FIX ============================

    grouped: Dict[str, Dict[str, Any]] = {}
    for r_ in rows:
        code = str(r_.get("I_CODE") or "").strip()
        name_ar = (r_.get("I_L_NAME") or r_.get("I_NAME") or "").strip()
        name_en = (r_.get("I_F_NAME") or "").strip() or name_ar
        key = f"{code}|{name_ar}|{name_en}"
        if key not in grouped:
            grouped[key] = {
                "id": code,
                "name_ar": name_ar,
                "name_en": name_en,
                "tax_percent": _to_float(r_.get("ITM_TAX_PRCNT"), DEFAULT_VAT_PERCENT),
                "variants": []
            }
        grouped[key]["variants"].append({
            "code": str(r_.get("P_SIZE") or "").strip(),
            "label": (r_.get("ITM_UNT") or "").strip() or "Default",
            "price": _to_float(r_.get("I_PRICE")),
        })
    for g in grouped.values():
        seen = set()
        uniq = []
        for v in g["variants"]:
            sig = (v["code"], v["label"], f"{_to_float(v['price']):.4f}")
            if sig not in seen:
                seen.add(sig)
                uniq.append(v)
        g["variants"] = uniq
    items = []
    for g in grouped.values():
        g["price"] = min((v["price"] for v in g["variants"]), default=0.0)
        items.append(g)
    return items


def _ensure_categories_loaded() -> List[Dict[str, Any]]:
    cats = st.session_state.get("categories")
    if not cats:
        st.session_state["categories"] = get_categories(_lang_code())
        cats = st.session_state["categories"] or []
    return cats


def ensure_items_loaded_for_cat(cat_id: str, lang_code: str):
    key = str(cat_id)
    if key not in st.session_state["items_by_cat"]:
        st.session_state["items_by_cat"][key] = get_items_for_group(cat_id, lang_code)


# ==================== STRICT NAME MATCHING ====================
def _tokens_norm(s: str, lang: str) -> List[str]:
    return _text_tokens(canonicalize(s or "", lang))


def _strict_name_match(query: str, name: str, lang: str) -> bool:
    q = canonicalize(query, lang)
    n = canonicalize(name, lang)
    if not q or not n: return False
    if q == n: return True
    qt = set(_tokens_norm(q, lang))
    nt = set(_tokens_norm(n, lang))
    if qt and qt.issubset(nt): return True
    ratio = difflib.SequenceMatcher(None, q, n).ratio()
    return ratio >= 0.85  # FIX: Was 0.90, now more flexible


# ==================== Persistence Helpers ====================
def _ensure_persist_sid():
    if "persist_sid" not in st.session_state:
        sid = st.query_params.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            st.query_params["sid"] = sid
        st.session_state["persist_sid"] = sid


def _sid_path() -> pathlib.Path:
    sid = st.session_state.get("persist_sid") or "default"
    return BASE_CACHE_DIR / f"{sid}.json"


def _save_state():
    try:
        data = {
            "messages": st.session_state.get("messages", []),
            "ai_messages": st.session_state.get("ai_messages", []),
            "cart": st.session_state.get("cart", []),
            "lang": st.session_state.get("lang", "ar"),
            "name": st.session_state.get("name", ""),
            "phone": st.session_state.get("phone", ""),
            "address": st.session_state.get("address", ""),
            "selected_category": st.session_state.get("selected_category"),
            "last_list": st.session_state.get("last_list", {}),
            "last_page_items": st.session_state.get("last_page_items", []),
            "last_page_categories": st.session_state.get("last_page_categories", []),
            "last_page_cart": st.session_state.get("last_page_cart", []),
        }
        _sid_path().write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_state_if_any():
    try:
        p = _sid_path()
        if not p.exists(): return False
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
        ss = st.session_state
        ss["messages"] = data.get("messages", [])
        ss["ai_messages"] = data.get("ai_messages", [])
        ss["cart"] = data.get("cart", [])
        ss["lang"] = data.get("lang", "ar")
        ss["_prev_lang"] = ss["lang"]
        ss["name"] = data.get("name", "")
        ss["phone"] = data.get("phone", "")
        ss["address"] = data.get("address", "")
        ss["selected_category"] = data.get("selected_category")
        ss["last_list"] = data.get("last_list", {})
        ss["last_page_items"] = data.get("last_page_items", [])
        ss["last_page_categories"] = data.get("last_page_categories", [])
        ss["last_page_cart"] = data.get("last_page_cart", [])
        return True
    except Exception:
        return False


_ensure_persist_sid()

# ==================== Session State Defaults ====================
ss = st.session_state
for k, v in [
    ("lang", "ar"), ("_prev_lang", "ar"), ("messages", []), ("cart", []),
    ("categories", None), ("flash", None), ("selected_category", None),
    ("items_cache", []), ("items_by_cat", {}), ("last_page_items", []),
    ("last_page_categories", []), ("last_list", {}), ("last_page_cart", []),
    ("name", ""), ("phone", ""), ("address", ""),
    ("ai_messages", []),
    ("voice_last_text", ""),
    ("voice_processing", False),
    ("_last_user_text", ""),
    ("_last_audio_hash", ""),
    (" _reset_mic", False),
    (" _mic_nonce", 0),
    (" _chat_nonce", 0),
    ("mic_mode", "Push to talk"),
    ("confirming_checkout", False), ("confirming_clear", False),
    ("voice_enabled", True), ("voice_choice", _tts_voice()),
    ("spoken_set", set()),
    ("sending_order", False),
    ("search_query", ""), ("active_category_id", None), ("qty_adder", 1),
    ("jarvis_active", False),
    ("persist_sid", ss.get("persist_sid", "")),
]:
    if k not in ss:
        ss[k] = v

_load_state_if_any()


# ==================== Variants & Cart Helpers ====================
def _choose_variant(item: Dict[str, Any],
                    size_label: Optional[str],
                    size_code: Optional[str]) -> Tuple[Optional[str], Optional[str], float]:
    variants = item.get("variants") or []
    if size_code is not None:
        for v in variants:
            if str(v.get("code")) == str(size_code):
                return (v.get("label"), str(v.get("code")), _to_float(v.get("price")))
    if size_label:
        q = canonicalize(size_label, "ar")
        for v in variants:
            if canonicalize(v.get("label") or "", "ar") == q:
                return (v.get("label"), str(v.get("code")), _to_float(v.get("price")))
        labs = [canonicalize(v.get("label") or "", "ar") for v in variants]
        m = difflib.get_close_matches(q, labs, n=1, cutoff=0.6)
        if m:
            for v in variants:
                if canonicalize(v.get("label") or "", "ar") == m[0]:
                    return (v.get("label"), str(v.get("code")), _to_float(v.get("price")))
    if variants:
        v0 = variants[0]
        return (v0.get("label"), str(v0.get("code")), _to_float(v0.get("price")))
    return (None, None, _to_float(item.get("price", 0)))


def merge_or_append_cart_line(item_id, label, unit_label, variant_code, unit_price, qty, notes, tax_percent):
    code_key = str(variant_code or "")
    for line in ss["cart"]:
        if str(line["id"]) == str(item_id) and str(line.get("variant_code") or "") == code_key:
            line["qty"] = int(_to_float(line.get("qty"), 0)) + int(_to_float(qty, 0))
            if notes:
                line["notes"] = f"{line.get('notes', '')}; {notes}".strip("; ").strip()
            _save_state()
            return
    ss["cart"].append({
        "id": item_id,
        "name": label,
        "price": unit_price,
        "qty": int(_to_float(qty, 1)),
        "size": unit_label,
        "variant_code": code_key,
        "notes": notes,
        "tax_percent": _to_float(tax_percent, DEFAULT_VAT_PERCENT),
    })
    _save_state()


def _cart_candidates_by_id(item_id):
    return [ln for ln in ss["cart"] if str(ln["id"]) == str(item_id)]


def _decrement_by_id_and_code(item_id, variant_code, qty, fallback_label=None):
    code_key = str(variant_code or "")
    remaining = int(_to_float(qty, 1))
    new_cart = []
    found_any = False

    for ln in ss["cart"]:
        same = (str(ln["id"]) == str(item_id)) and (str(ln.get("variant_code") or "") == code_key)
        if same and remaining > 0:
            found_any = True
            cur = int(_to_float(ln.get("qty"), 1))
            if cur > remaining:
                ln["qty"] = cur - remaining
                remaining = 0
                new_cart.append(ln)
            elif cur < remaining:
                remaining -= cur
            else:
                remaining = 0
        else:
            new_cart.append(ln)

    if not found_any and fallback_label:
        rem2 = int(_to_float(qty, 1))
        new2 = []
        for ln in new_cart:
            same = (str(ln["id"]) == str(item_id)) and ((ln.get("size") or "") == (fallback_label or ""))
            if same and rem2 > 0:
                cur = int(_to_float(ln.get("qty"), 1))
                if cur > rem2:
                    ln["qty"] = cur - rem2
                    rem2 = 0
                    new2.append(ln)
                elif cur < rem2:
                    rem2 -= cur
                else:
                    rem2 = 0
            else:
                new2.append(ln)
        new_cart, remaining = new2, rem2

    ss["cart"] = new_cart
    _save_state()
    return {"ok": True, "remaining": remaining}


def decrement_or_remove_cart_line_by_item(item: Dict[str, Any],
                                          qty: int,
                                          size_label: Optional[str] = None,
                                          size_code: Optional[str] = None):
    label, code, _ = _choose_variant(item, size_label, size_code)

    if code is None:
        cands = _cart_candidates_by_id(item["id"])
        if not cands:
            return {"ok": False, "error": "not_in_cart"}
        codes = list({str(ln.get("variant_code") or "") for ln in cands})
        if len(codes) == 1:
            code = codes[0]
            label = next((ln.get("size") for ln in cands), label)
        else:
            return {"ok": False, "error": "ambiguous_size",
                    "sizes": [ln.get("size") or "" for ln in cands],
                    "codes": codes}

    return _decrement_by_id_and_code(item["id"], code, qty, fallback_label=label)


# ==================== Matching & Finders ====================
def _best_item_match(items_all: List[Dict[str, Any]], q: str, lang: str) -> Optional[Dict[str, Any]]:
    qn = canonicalize(q, lang)
    if not qn: return None
    for it in items_all:
        name = it["name_ar"] if lang == "ar" else it["name_en"]
        if _strict_name_match(q, name, lang):
            return it
    return None


def quick_candidate_categories(user_text: str, categories: List[Dict[str, Any]], lang: str, max_cats: int = 3) -> List[
    Dict[str, Any]]:
    toks = set(_text_tokens(user_text));
    scored = []
    for c in categories:
        nm = canonicalize(c["name_ar"] if lang == "ar" else c["name_en"], lang)
        scored.append((sum(1 for tok in toks if tok and tok in nm), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [c for s, c in scored if s > 0][:max_cats]
    return out or [c for _, c in scored[:max_cats]]


def find_item_anywhere(query_name: str, lang: str, lang_code: str, categories: List[Dict[str, Any]],
                       utterance_hint: str = "") -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    _ensure_categories_loaded()
    if ss.get("items_cache"):
        it = _best_item_match(ss["items_cache"], query_name, lang)
        if it and ss.get("selected_category"):
            return it, ss["selected_category"]
    for cat_id, items in (ss.get("items_by_cat") or {}).items():
        it = _best_item_match(items, query_name, lang)
        if it:
            for c in ss["categories"]:
                if str(c["id"]) == str(cat_id): return it, c
    cats = ss["categories"] or categories or []
    hints = quick_candidate_categories(utterance_hint or query_name, cats, lang, max_cats=6) or cats[:6]
    for cat in hints:
        ensure_items_loaded_for_cat(cat["id"], lang_code)
        it = _best_item_match(ss["items_by_cat"][str(cat["id"])], query_name, lang)
        if it: return it, cat
    return None


def find_item_by_code(item_code: str, lang_code: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    _ensure_categories_loaded()
    code = str(item_code).strip()
    for cat_id, items in (ss.get("items_by_cat") or {}).items():
        for it in items:
            if str(it.get("id")) == code:
                cat = next((c for c in ss.get("categories") or [] if str(c["id"]) == str(cat_id)), None)
                return it, cat
    for cat in (ss.get("categories") or []):
        ensure_items_loaded_for_cat(cat["id"], lang_code)
        for it in ss["items_by_cat"][str(cat["id"])]:
            if str(it.get("id")) == code:
                return it, cat
    return None


# ==================== Tools (Bridge) & AI helpers ====================
def tool_set_language(lang: str):
    if lang not in ("ar", "en"): return {"ok": False, "error": "unsupported_language"}
    apply_language(lang)
    _save_state()
    return {"ok": True, "lang": lang}


def tool_list_categories(offset: int = 0, limit: int = 20):
    cats = _ensure_categories_loaded()
    slice_ = cats[offset: offset + limit]
    ss["last_page_categories"] = list(slice_)
    ss["last_list"] = {"type": "categories"}
    _save_state()
    return {
        "total": len(cats),
        "offset": offset,
        "limit": limit,
        "categories": [
            {"id": c["id"], "name": (c["name_ar"] if ss["lang"] == "ar" else c["name_en"])}
            for c in slice_
        ]
    }


def filter_items(items: List[Dict[str, Any]], q: Optional[str], lang: str) -> List[Dict[str, Any]]:
    if not q: return items
    qn = canonicalize(q, lang)
    out: List[Dict[str, Any]] = []
    for it in items:
        nm = canonicalize(it["name_ar"] if lang == "ar" else it["name_en"], lang)
        if qn in nm or nm in qn:
            out.append(it);
            continue
        for v in it.get("variants", []):
            lab = canonicalize(v.get("label") or "", lang)
            if qn in lab or lab in qn:
                out.append(it);
                break
    if out: return out
    names = [canonicalize(it["name_ar"] if lang == "ar" else it["name_en"], lang) for it in items]
    for name in difflib.get_close_matches(qn, names, n=10, cutoff=0.6):
        for it in items:
            nm = canonicalize(it["name_ar"] if lang == "ar" else it["name_en"], lang)
            if nm == name: out.append(it); break
    return out or items


def tool_list_items(category_id: str, offset: int = 0, limit: int = 25, search: Optional[str] = None):
    ensure_items_loaded_for_cat(category_id, _lang_code())
    items = ss["items_by_cat"][str(category_id)]
    items = filter_items(items, search, ss["lang"])
    slice_ = items[offset: offset + limit]
    ss["last_page_items"] = list(slice_)
    ss["selected_category"] = next((c for c in (ss.get("categories") or []) if str(c["id"]) == str(category_id)), None)
    ss["items_cache"] = list(items)
    ss["last_list"] = {"type": "items"}
    _save_state()

    category_name = (ss["selected_category"]["name_ar"] if ss["lang"] == "ar" and ss["selected_category"] else
                     (ss["selected_category"]["name_en"] if ss["selected_category"] else ""))

    return {
        "category_id": category_id,
        "category_name": category_name,
        "total": len(items),
        "offset": offset,
        "limit": limit,
        "items": [{
            "id": it["id"],
            "name": (it["name_ar"] if ss["lang"] == "ar" else it["name_en"]),
            "min_price": _to_float(it.get("price", 0)),
            "variants": [{"code": v.get("code"), "label": v.get("label"), "price": _to_float(v.get("price", 0))} for v
                         in it.get("variants", [])]
        } for it in slice_]
    }


def _pool_items_for_search(max_cats: int = 8):
    cats = _ensure_categories_loaded()
    pool = list(ss.get("items_cache") or [])
    chosen = cats[:max_cats]
    for c in chosen:
        ensure_items_loaded_for_cat(c["id"], _lang_code())
        pool.extend(ss["items_by_cat"][str(c["id"])])
    seen = set()
    uniq = []
    for it in pool:
        sid = str(it["id"])
        if sid not in seen:
            seen.add(sid)
            uniq.append(it)
    return uniq


def tool_search_items(query: str, limit: int = 15):
    # === FIX START: First, try to find a matching category by name ===
    cats = _ensure_categories_loaded()
    lang = ss["lang"]
    q_canon = canonicalize(query, lang)

    matched_cat = None
    for c in cats:
        cat_name = canonicalize(c["name_ar"] if lang == "ar" else c["name_en"], lang)
        if q_canon in cat_name:
            matched_cat = c
            break

    if matched_cat:
        return tool_list_items(category_id=str(matched_cat["id"]), limit=limit)
    # === FIX END: If no category matches, continue with the old logic ===

    pool = _pool_items_for_search()
    qn = canonicalize(query, ss["lang"])

    def score(it):
        name = canonicalize(it["name_ar"] if ss["lang"] == "ar" else it["name_en"], ss["lang"])
        a = set(_text_tokens(qn));
        b = set(_text_tokens(name))
        inter = len(a & b);
        uni = len(a | b) or 1
        base = inter / uni
        sub = 1.0 if (qn in name or name in qn) else 0.0
        return base + 0.5 * sub

    pool.sort(key=score, reverse=True)
    out = pool[:max(1, min(limit, 25))]
    ss["last_page_items"] = list(out)
    ss["last_list"] = {"type": "items"}
    _save_state()
    return {
        "query": query,
        "items": [{
            "id": it["id"],
            "name": (it["name_ar"] if ss["lang"] == "ar" else it["name_en"]),
            "min_price": _to_float(it.get("price", 0)),
            "variants": [{"code": v.get("code"), "label": v.get("label"), "price": _to_float(v.get("price", 0))} for v
                         in it.get("variants", [])]
        } for it in out]
    }


# ==================== Cart Tooling ====================
def tool_add_item_by_index(index: int, qty: int = 1, size_label: Optional[str] = None, size_code: Optional[str] = None,
                           notes: Optional[str] = None):
    lst = ss.get("last_page_items") or []
    if not lst or index < 1 or index > len(lst):
        return {"ok": False, "error": "index_out_of_range", "shown_count": len(lst)}
    it = lst[index - 1]
    unit_label, variant_code, unit_price = _choose_variant(it, size_label, size_code)
    label = it["name_ar"] if ss["lang"] == "ar" else it["name_en"]
    merge_or_append_cart_line(it["id"], label, unit_label, variant_code, unit_price, qty, notes,
                              it.get("tax_percent", DEFAULT_VAT_PERCENT))
    return {"ok": True,
            "added": {"id": it["id"], "name": label, "qty": qty, "size": unit_label, "size_code": variant_code,
                      "unit_price": unit_price}}


def tool_add_item_by_name(name: str, qty: int = 1, size_label: Optional[str] = None, size_code: Optional[str] = None,
                          notes: Optional[str] = None):
    _ensure_categories_loaded()
    found = find_item_anywhere(name, ss["lang"], _lang_code(), ss.get("categories") or [], name)
    if not found:
        pool = _pool_items_for_search()
        qn = canonicalize(name, ss["lang"])

        def score(it):
            nm = canonicalize(it["name_ar"] if ss["lang"] == "ar" else it["name_en"], ss["lang"])
            a = set(_text_tokens(qn));
            b = set(_text_tokens(nm))
            inter = len(a & b);
            uni = len(a | b) or 1
            base = inter / uni
            sub = 1.0 if (qn in nm or nm in qn) else 0.0
            starts = 0.2 if nm.startswith(qn) else 0.0
            return base + 0.5 * sub + starts

        pool.sort(key=score, reverse=True)
        sugg = [{
            "id": it["id"],
            "name": (it["name_ar"] if ss["lang"] == "ar" else it["name_en"]),
            "min_price": _to_float(it.get("price", 0)),
            "variants": [{"code": v.get("code"), "label": v.get("label"),
                          "price": _to_float(v.get("price", 0))} for v in it.get("variants", [])]
        } for it in pool[:8]]
        return {"ok": False, "error": "not_found", "suggestions": sugg}

    match, _src_cat = found
    match_name = match["name_ar"] if ss["lang"] == "ar" else match["name_en"]
    if not _strict_name_match(name, match_name, ss["lang"]):
        pool = _pool_items_for_search()
        qn = canonicalize(name, ss["lang"])

        def score(it):
            nm = canonicalize(it["name_ar"] if ss["lang"] == "ar" else it["name_en"], ss["lang"])
            a = set(_text_tokens(qn));
            b = set(_text_tokens(nm))
            inter = len(a & b);
            uni = len(a | b) or 1
            base = inter / uni
            sub = 1.0 if (qn in nm or nm in qn) else 0.0
            starts = 0.2 if nm.startswith(qn) else 0.0
            return base + 0.5 * sub + starts

        pool.sort(key=score, reverse=True)
        sugg = [{
            "id": it["id"],
            "name": (it["name_ar"] if ss["lang"] == "ar" else it["name_en"]),
            "min_price": _to_float(it.get("price", 0)),
            "variants": [{"code": v.get("code"), "label": v.get("label"),
                          "price": _to_float(v.get("price", 0))} for v in it.get("variants", [])]
        } for it in pool[:8]]
        return {"ok": False, "error": "not_found", "suggestions": sugg}

    unit_label, variant_code, unit_price = _choose_variant(match, size_label, size_code)
    label = match["name_ar"] if ss["lang"] == "ar" else match["name_en"]
    merge_or_append_cart_line(match["id"], label, unit_label, variant_code, unit_price, qty, notes,
                              match.get("tax_percent", DEFAULT_VAT_PERCENT))
    return {"ok": True, "added": {"id": match["id"], "name": label, "qty": qty,
                                  "size": unit_label, "size_code": variant_code, "unit_price": unit_price}}


def tool_add_item_by_code(code: str, qty: int = 1, size_label: Optional[str] = None,
                          size_code: Optional[str] = None, notes: Optional[str] = None):
    found = find_item_by_code(code, _lang_code())
    if not found:
        return {"ok": False, "error": "not_found"}
    match, _src_cat = found
    unit_label, variant_code, unit_price = _choose_variant(match, size_label, size_code)
    label = match["name_ar"] if ss["lang"] == "ar" else match["name_en"]
    merge_or_append_cart_line(match["id"], label, unit_label, variant_code, unit_price, qty, notes,
                              match.get("tax_percent", DEFAULT_VAT_PERCENT))
    return {"ok": True, "added": {"id": match["id"], "name": label, "qty": qty,
                                  "size": unit_label, "size_code": variant_code, "unit_price": unit_price}}


def tool_remove_item_by_index(index: int, qty: int = 1, size_label: Optional[str] = None,
                              size_code: Optional[str] = None):
    lst = ss.get("last_page_items") or []
    if not lst or index < 1 or index > len(lst):
        return {"ok": False, "error": "index_out_of_range", "shown_count": len(lst)}
    it = lst[index - 1]
    res = decrement_or_remove_cart_line_by_item(it, qty, size_label=size_label, size_code=size_code)
    if not res.get("ok"):
        res["item_name"] = it["name_ar"] if ss["lang"] == "ar" else it["name_en"]
        return res
    label = it["name_ar"] if ss["lang"] == "ar" else it["name_en"]
    return {"ok": True,
            "removed": {"id": it["id"], "name": label, "qty": qty, "size_label": size_label, "size_code": size_code}}


def tool_remove_item_by_name(name: str, qty: int = 1, size_label: Optional[str] = None,
                             size_code: Optional[str] = None):
    # === FIX START: Search directly within the cart, not the entire menu ===
    lang = ss.get("lang", "ar")
    q_canon = canonicalize(name, lang)

    cart_candidates = []
    for i, line in enumerate(ss["cart"]):
        line_name_canon = canonicalize(line.get("name", ""), lang)
        # Use a flexible ratio to find the best match in the cart
        ratio = difflib.SequenceMatcher(None, q_canon, line_name_canon).ratio()
        if ratio > 0.75:
            cart_candidates.append((ratio, line))

    if not cart_candidates:
        return {"ok": False, "error": "not_found_in_cart"}

    # Find the best match from the candidates
    cart_candidates.sort(key=lambda x: x[0], reverse=True)
    best_match_line = cart_candidates[0][1]

    item_id = best_match_line.get("id")
    variant_code = best_match_line.get("variant_code")

    res = _decrement_by_id_and_code(item_id, variant_code, qty, fallback_label=best_match_line.get("size"))

    if not res.get("ok"):
        res["item_name"] = best_match_line.get("name")
        return res

    return {"ok": True, "removed": {"id": item_id, "name": best_match_line.get("name"), "qty": qty}}
    # === FIX END ===


def tool_show_cart():
    ss["last_page_cart"] = [
        {"id": it["id"], "name": it["name"], "qty": it["qty"], "size": it.get("size"),
         "variant_code": it.get("variant_code"), "unit_price": _to_float(it.get("price", 0))}
        for it in ss["cart"]
    ]
    ss["last_list"] = {"type": "cart"}
    _save_state()
    return {
        "items": ss["last_page_cart"],
        "total": cart_total(ss["cart"]),
        "currency": CURRENCY
    }


def tool_clear_cart():
    ss["cart"] = []
    _save_state()
    return {"ok": True}


def _mask_order_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    masked_payload = json.loads(json.dumps(payload))
    try:
        mst = masked_payload.get("Value", {}).get("RMS_CSTMR_ORDR", {}).get("RMS_CSTMR_ORDR_MST", {})
        if "MOBILE_NO" in mst and mst["MOBILE_NO"]:
            phone = str(mst["MOBILE_NO"])
            if len(phone) > 5:
                mst["MOBILE_NO"] = f"{phone[:3]}***{phone[-2:]}"
            else:
                mst["MOBILE_NO"] = "***"
    except Exception:
        pass
    return masked_payload


def insert_customer_order(ui: Dict[str, Any]) -> Dict[str, Any]:
    if not ORDER_URL:
        return {"ok": False, "error": "ORDER_URL missing in .env",
                "raw": None, "status": None, "text": None, "order_id": None}

    lang_no = int(ui.get("lang_code", LANG_CODE_AR))
    vat_default = DEFAULT_VAT_PERCENT
    details = []

    for line in ui.get("items", []):
        qty = int(_to_float(line.get("qty"), 1))
        unit_price = _to_float(line.get("unit_price"), 0.0)
        vat_percent = _to_float(line.get("tax_percent"), vat_default)

        net, gross, vat_amt = split_price_by_vat(unit_price, vat_percent)
        item_id = line.get("item_id") or line.get("id")
        try:
            item_id = int(item_id)
        except Exception:
            pass

        p_size = line.get("variant_code") or 1
        try:
            p_size = int(p_size)
        except Exception:
            p_size = 1

        details.append({
            "ITM_UNT": line.get("size"),
            "I_CODE": item_id,
            "I_PRICE": _r2(net),
            "I_PRICE_VAT": _r2(gross),
            "I_QTY": qty,
            "P_SIZE": p_size,
            "VAT_AMT": _r2(vat_amt),
            "VAT_PRCNT": float(vat_percent),
            "ITM_DISC_PRCNT": float(_to_float(line.get("discount_percent", 0))),
            "ITM_ATTCH_FLG": 2,
            "itemName": line.get("name"),
        })

    mst = {
        "ADDRSS_NO": "", "BILL_DOC_TYPE": BILL_DOC_TYPE, "BRN_NO": P_BRN_NO,
        "DEVIC_TYP": DEVIC_TYP, "OTHR_AMT": ui.get("other_amount", ""), "APP_TYP": APP_TYP,
        "CLC_TBL_SRVC_FLG": CLC_TBL_SRVC_FLG, "NOT_CLC_SRVC_TYP": NOT_CLC_SRVC_TYP,
        "CSTMR_RGN_NO": ui.get("region_no", ""), "DLVRY_AMT": ui.get("delivery_amount", ""),
        "LNG_NO": lang_no, "MOBILE_NO": (ui.get("customer") or {}).get("phone") or "",
        "PYMNT_FLG": PYMNT_FLG, "RES_UNRES_ALL": RES_UNRES_ALL, "FRM_MNU_APP_FLG": FRM_MNU_APP_FLG
    }

    body = {
        "Value": {
            "P_LANG_NO": lang_no, "P_DVS_TYP": P_DVS_TYP, "P_BRN_NO": P_BRN_NO,
            "CREDIT_GROUP_TYP": CREDIT_GROUP_TYP,
            "RMS_CSTMR_ORDR": {
                "RMS_CSTMR_ORDR_DTL": details, "RMS_CSTMR_ORDR_MST": mst
            }
        }
    }
    ss["last_order_request"] = _mask_order_payload(body)
    _save_state()

    if DEBUG_ORDERS:
        st.info("Sending order payload …")
        st.code(json.dumps(ss["last_order_request"], ensure_ascii=False, indent=2))

    try:
        r = _post_with_retries(ORDER_URL, body, timeout=ORDER_TIMEOUT)
        status, text = r.status_code, r.text
        resp_json = None
        try:
            resp_json = r.json() or {}
        except Exception:
            pass

        ss["last_order_response"] = {"status": status, "text": text, "json": resp_json}
        _save_state()

        if status >= 400:
            if DEBUG_ORDERS: st.error(f"HTTP {status}"); st.code(text)
            return {"ok": False, "error": f"HTTP {status}", "raw": None, "status": status, "text": text,
                    "order_id": None}

        if resp_json is None:
            if DEBUG_ORDERS: st.error("Non-JSON response:"); st.code(text)
            return {"ok": False, "error": "Non-JSON response", "raw": None, "status": status, "text": text,
                    "order_id": None}

        result = resp_json.get("Result") or {}
        ok = str(result.get("ErrNo")) == "0"
        order_id = result.get("DocSrl") or result.get("DocNo") or result.get("AddValue2")

        if DEBUG_ORDERS: st.success(f"Order response (ok={ok})"); st.code(
            json.dumps(resp_json, ensure_ascii=False, indent=2))
        return {"ok": ok, "order_id": order_id, "raw": resp_json, "status": status, "text": text}
    except requests.exceptions.Timeout:
        ss["last_order_response"] = {"status": None, "text": f"Timeout after {ORDER_TIMEOUT}s", "json": None};
        _save_state()
        return {"ok": False, "error": f"Timeout after {ORDER_TIMEOUT}s", "raw": None, "status": None, "text": None,
                "order_id": None}
    except Exception as e:
        ss["last_order_response"] = {"status": None, "text": f"Unexpected error: {e}", "json": None};
        _save_state()
        return {"ok": False, "error": f"Unexpected error: {e}", "raw": None, "status": None, "text": None,
                "order_id": None}


def tool_checkout(phone: Optional[str] = None, address: Optional[str] = None):
    phone = _normalize_phone(phone or ss.get("phone"))
    address = address or ss.get("address", "")
    if not phone: return {"ok": False, "error": "missing_phone"}
    payload = {"customer": {"name": ss.get("name", ""), "phone": phone},
               "fulfillment": "delivery" if address else "pickup", "address": address,
               "items": [{"item_id": it["id"], "name": it["name"], "qty": it["qty"], "unit_price": it["price"],
                          "size": it.get("size"),
                          "variant_code": it.get("variant_code"),
                          "tax_percent": it.get("tax_percent", DEFAULT_VAT_PERCENT)} for it in ss["cart"]],
               "total": cart_total(ss["cart"]), "currency": CURRENCY, "lang_code": _lang_code()}
    resp = insert_customer_order(payload)
    if resp.get("ok"):
        ss["cart"] = [];
        _save_state()
        return {"ok": True, "order_id": resp.get("order_id")}
    _save_state()
    return {"ok": False, "error": resp.get("error") or "order_failed", "details": resp.get("raw")}


def tool_remove_from_cart_by_index(index: int, qty: int = 1):
    lst = ss.get("last_page_cart") or []
    if not lst or index < 1 or index > len(lst):
        return {"ok": False, "error": "index_out_of_range", "shown_count": len(lst), "context": "cart"}
    line = lst[index - 1]
    item_id = line.get("id");
    variant_code = line.get("variant_code");
    size_label = line.get("size")
    res = _decrement_by_id_and_code(item_id, variant_code, qty, fallback_label=size_label)
    if not res.get("ok"): return {"ok": False, "error": "remove_failed", "details": res}
    tool_show_cart()
    return {"ok": True, "removed": {"id": item_id, "qty": qty, "index": index}}


def tool_select_number(index: int, qty: int = 1):
    info = ss.get("last_list") or {}
    if not info: return {"ok": False, "error": "no_last_list"}
    if info.get("type") == "categories":
        cats = ss.get("last_page_categories") or []
        if not cats or index < 1 or index > len(cats):
            return {"ok": False, "error": "index_out_of_range", "shown_count": len(cats), "context": "categories"}
        cat = cats[index - 1]
        return tool_list_items(category_id=str(cat["id"]), offset=0, limit=25)
    if info.get("type") == "items": return tool_add_item_by_index(index=index, qty=qty)
    if info.get("type") == "cart": return tool_remove_from_cart_by_index(index=index, qty=qty)
    return {"ok": False, "error": "unknown_last_list_type"}


# ==================== Language Apply & CSS ====================
def apply_language(new_lang: str):
    ss["lang"] = ss["_prev_lang"] = new_lang
    ss["categories"] = None;
    ss["items_by_cat"] = {};
    ss["items_cache"] = []
    ss["selected_category"] = None;
    ss["active_category_id"] = None
    ss["last_page_items"] = [];
    ss["last_page_categories"] = [];
    ss["last_list"] = {}
    _inject_css()


def _inject_css():
    rtl_styles = "direction: rtl; text-align: right;" if ss.get("lang") == "ar" else "direction: ltr; text-align: left;"
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;700&family=Manrope:wght@400;600;800&display=swap');

        body {{
            font-family: 'Manrope', 'IBM Plex Sans Arabic', sans-serif;
        }}
        .stApp {{
            {rtl_styles}
            background-color: #f8fafc;
        }}

        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }}

        .message-container {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
            gap: 10px;
        }}
        .user-message .message-bubble {{
            background-color: #cffafe;
            color: #083344;
            border-radius: 20px 20px 5px 20px;
        }}
        .bot-message .message-bubble {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px 20px 20px 5px;
        }}
        .message-bubble {{
            padding: 12px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            max-width: 85%;
            word-wrap: break-word;
        }}
        .message-avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }}
        .user-avatar {{ background-color: #67e8f9; }}
        .bot-avatar {{ background-color: #e2e8f0; }}

        .cart-panel {{
            background-color: #ffffff;
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            position: sticky;
            top: 55px;
        }}
        .cart-line {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #f1f5f9;
            font-size: 0.95rem;
        }}
        .cart-line span:first-child {{ font-weight: 500; color: #334155; }}
        .cart-line span:last-child {{ font-weight: 600; color: #0f172a; }}
        .cart-total {{
            font-weight: 800; font-size: 1.2rem; margin-top: 1rem;
            padding-top: 1rem; border-top: 2px solid #cbd5e1;
            display: flex; justify-content: space-between;
        }}
        .empty-cart-text {{ color: #64748b; text-align: center; padding: 2rem 0; }}

        [data-testid="stChatInput"] {{
            background-color: #ffffff;
        }}
    </style>
    """, unsafe_allow_html=True)
# --- WebRTC-based JARVIS (works on Streamlit Cloud) ---
def _downsample_48k_to_16k_int16(x: np.ndarray) -> np.ndarray:
    return x[::3].astype(np.int16)

class PorcupineProcessor(AudioProcessorBase):
    def __init__(self):
        self.ready = False
        self.porcupine = None
        self.frame_len = None
        self.buffer = np.array([], dtype=np.int16)

        try:
            if not PICOVOICE_API_KEY:
                self.err = "PICOVOICE_API_KEY missing"
                return
            import pvporcupine
            self.porcupine = pvporcupine.create(access_key=PICOVOICE_API_KEY, keywords=["jarvis"])
            self.frame_len = self.porcupine.frame_length
            self.ready = True
            self.err = None
        except Exception as e:
            self.err = f"Porcupine init failed: {e}"

    def recv_audio(self, frames):
        if not self.ready:
            return frames

        try:
            pcm_all = np.zeros(0, dtype=np.int16)
            for f in frames:
                ch = f.to_ndarray(format="s16").astype(np.int16)
                if ch.ndim == 2 and ch.shape[0] > 1:  # stereo → mono
                    ch = ch.mean(axis=0).astype(np.int16)
                pcm_all = np.concatenate([pcm_all, ch])

            pcm_16k = _downsample_48k_to_16k_int16(pcm_all)

            self.buffer = np.concatenate([self.buffer, pcm_16k])
            while len(self.buffer) >= self.frame_len:
                frame = self.buffer[:self.frame_len]
                self.buffer = self.buffer[self.frame_len:]
                res = self.porcupine.process(frame.tolist())
                if res >= 0:
                    st.session_state["jarvis_triggered_web"] = True
                    break
        except Exception as e:
            st.session_state["jarvis_webrtc_error"] = str(e)

        return frames

def start_web_jarvis():
    st.info("🎧 Web JARVIS is listening in your browser (say 'Jarvis').")
    ctx = webrtc_streamer(
        key="jarvis-webrtc",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=PorcupineProcessor,
    )
    return ctx


apply_language(ss["lang"])
_save_state()

# ==================== Header & Sidebar ====================
st.title("🤖 Onyx AI Chatbot")
st.caption(f"Session ID: {ss.get('persist_sid')[:8]}…")

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("👤 User Details")
    old_name, old_phone = ss.get("name", ""), ss.get("phone", "")
    ss["name"] = st.text_input("Name / الاسم", value=ss.get("name", ""))
    ss["phone"] = st.text_input("Phone / الجوال", value=ss.get("phone", ""))
    if ss["name"] != old_name or ss["phone"] != old_phone: _save_state()

    st.subheader("🌐 Language & Voice")

    st.subheader("JARVIS (Web)")
    if st.button("Start JARVIS (Web)"):
        st.session_state["jarvis_web_active"] = True
        st.session_state["jarvis_triggered_web"] = False
        st.rerun()

    if st.session_state.get("jarvis_web_active"):
        start_web_jarvis()
        if st.session_state.get("jarvis_webrtc_error"):
            st.error(f"WebRTC error: {st.session_state['jarvis_webrtc_error']}")
        if st.session_state.get("jarvis_triggered_web"):
            st.success("✔️ Wake word detected (Web)!")
            # TODO: after detection, trigger mic_recorder or your command capture flow
    mic_options = ['Push to talk', 'JARVIS Mode']
    ss["mic_mode"] = st.selectbox(
        "Microphone Mode / وضع الميكروفون",
        options=mic_options,
        index=mic_options.index(ss.get('mic_mode', 'Push to talk'))
    )
    prev_lang = ss.get("lang", "ar")
    lang_choice = st.selectbox("Language / اللغة", ["ar", "en"], index=0 if ss["lang"] == "ar" else 1)
    if lang_choice != prev_lang:
        apply_language(lang_choice)
        _save_state()
        st.rerun()

    ss["voice_enabled"] = st.toggle("🔊 Speak replies", value=ss.get("voice_enabled", True))
    ss["voice_choice"] = st.selectbox("Voice", ["alloy", "verse", "coral", "sage", "breeze", "flow"],
                                      index=(["alloy", "verse", "coral", "sage", "breeze", "flow"].index(
                                          ss.get("voice_choice", _tts_voice()))
                                             if ss.get("voice_choice", _tts_voice()) in ["alloy", "verse", "coral",
                                                                                         "sage", "breeze",
                                                                                         "flow"] else 0))

    st.subheader("🛠️ Developer")
    if st.button("Clear Cache & Reset Session", use_container_width=True):
        st.cache_data.clear()
        keys_to_clear = [
            "messages", "ai_messages", "cart",
            "last_list", "last_page_items", "last_page_categories", "last_page_cart"
        ]
        for key in keys_to_clear:
            if key in ss:
                del ss[key]
        try:
            session_file = _sid_path()
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            st.warning(f"Could not delete session file: {e}")

        st.toast("Cache and session have been cleared!", icon="✅")
        st.rerun()

# ==================== System Prompt ====================
SYSTEM_PROMPT = """
You are Onyx, a warm, human-like restaurant assistant. Speak naturally in the user's language (Arabic or English).
- Keep replies concise for voice. Use short sentences.
- When you call the `list_items` tool, it will return a `category_name`. You MUST use this exact `category_name` as the title for the list you show the user. For example, say "Here is the [category_name] menu:". This is critical to avoid confusion.
- If an item is not found, apologize and call `search_items` to suggest similar ones.

- **CRITICAL INSTRUCTION for Handling Numbers:**
  - When a user provides a number (e.g., "1", "add 5", "number 18"), it **ALWAYS** refers to the item's **position (index)** in the most recent list you displayed. It is NEVER an item ID.
  - You **MUST** use the `select_number` tool to handle these inputs. Do not try to guess the item name or use a different tool.
  - **Example 1:** If you just showed categories and the user says "18", you MUST call `select_number(index=18)`.
  - **Example 2:** If you just showed a list of items and the user says "add number 2", you MUST call `select_number(index=2)`.

- If multiple sizes for an item are available, **ask the user** which one to choose.
- Always **confirm before placing an order**: Show the cart contents and total, then ask for final confirmation.
- If a user asks to clear the cart (e.g., "cancel order", "clear cart"), you must ask for confirmation before calling the `clear_cart` tool.
- **Ask for phone/address only** when the user agrees to checkout and those details are missing.
"""


# ==================== OpenAI Chat Loop ====================
def _call_tool(name: str, args: dict):
    try:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        if name == "set_language": return tool_set_language(**args)
        if name == "list_categories": return tool_list_categories(**args)
        if name == "list_items": return tool_list_items(**args)
        if name == "search_items": return tool_search_items(**args)
        if name == "add_item_by_index": return tool_add_item_by_index(**args)
        if name == "add_item_by_name": return tool_add_item_by_name(**args)
        if name == "add_item_by_code": return tool_add_item_by_code(**args)
        if name == "remove_item_by_index": return tool_remove_item_by_index(**args)
        if name == "remove_item_by_name": return tool_remove_item_by_name(**args)
        if name == "select_number": return tool_select_number(**args)
        if name == "remove_from_cart_by_index": return tool_remove_from_cart_by_index(**args)
        if name == "show_cart": return tool_show_cart()
        if name == "checkout": return tool_checkout(**args)
        if name == "clear_cart": return tool_clear_cart()
        return {"ok": False, "error": "unknown_tool"}
    except Exception as e:
        return {"ok": False, "error": f"exception: {e}"}


def _current_lang(): return ss.get("lang", "ar")


def _lang_code():   return LANG_CODE_AR if ss.get("lang", "ar") == "ar" else LANG_CODE_EN


def _trim_history(max_msgs=40):
    if len(ss["ai_messages"]) > max_msgs:
        ss["ai_messages"] = [ss["ai_messages"][0]] + ss["ai_messages"][-(max_msgs - 1):]
        _save_state()


def agent_reply(user_text: str) -> str:
    # === FIX: Robustly manage the AI message history and system prompt ===
    # Clean any non-dictionary items from the history, just in case
    ss["ai_messages"] = [m for m in ss.get("ai_messages", []) if isinstance(m, dict)]

    # If the history is empty or doesn't start with a system message, create a new one.
    if not ss["ai_messages"] or ss["ai_messages"][0].get("role") != "system":
        ss["ai_messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Always update the system prompt with the current language to ensure it's up-to-date.
    ss["ai_messages"][0]["content"] = SYSTEM_PROMPT + f"\nCurrent app language: {_current_lang()}"
    # ============================ END OF FIX ============================

    ss["ai_messages"].append({"role": "user", "content": user_text})

    for _ in range(8):
        try:
            r = ai().chat.completions.create(
                model=OPENAI_MODEL, temperature=0.6, messages=ss["ai_messages"],
                tools=TOOLS, tool_choice="auto"
            )
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            # Remove the last user message to allow a retry
            ss["ai_messages"].pop()
            return "Sorry, I encountered an error. Please try again."

        msg = r.choices[0].message
        calls = getattr(msg, "tool_calls", None)

        if calls:
            ss["ai_messages"].append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in
                               calls]
            })
            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                result = _call_tool(name, args)
                ss["ai_messages"].append({
                    "role": "tool", "tool_call_id": c.id, "name": name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            _trim_history()
            _save_state()
            continue

        text = (msg.content or "").strip()
        if not text:
            # If the model returns nothing, provide a generic response
            text = "Is there anything else I can help with?"

        ss["ai_messages"].append({"role": "assistant", "content": text})
        _trim_history()
        _save_state()
        return text

    fallback = "I seem to be stuck. Could you please rephrase your request?"
    ss["ai_messages"].append({"role": "assistant", "content": fallback})
    _trim_history()
    _save_state()
    return fallback


TOOLS = [
    {"type": "function",
     "function": {"name": "set_language", "description": "Switch assistant language to 'ar' or 'en'.",
                  "parameters": {"type": "object", "properties": {"lang": {"type": "string", "enum": ["ar", "en"]}},
                                 "required": ["lang"]}}},
    {"type": "function", "function": {"name": "list_categories", "description": "List menu categories with pagination.",
                                      "parameters": {"type": "object",
                                                     "properties": {"offset": {"type": "integer", "default": 0},
                                                                    "limit": {"type": "integer", "default": 20}}}}},
    {"type": "function", "function": {"name": "list_items",
                                      "description": "List items for a category with pagination and optional text filter.",
                                      "parameters": {"type": "object", "properties": {"category_id": {"type": "string"},
                                                                                      "offset": {"type": "integer",
                                                                                                 "default": 0},
                                                                                      "limit": {"type": "integer",
                                                                                                "default": 25},
                                                                                      "search": {"type": "string"}},
                                                     "required": ["category_id"]}}},
    {"type": "function", "function": {"name": "search_items", "description": "Search items globally across categories.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                                                      "limit": {"type": "integer",
                                                                                                "default": 15}},
                                                     "required": ["query"]}}},
    {"type": "function",
     "function": {"name": "add_item_by_index", "description": "Add item using its index from the last shown list.",
                  "parameters": {"type": "object",
                                 "properties": {"index": {"type": "integer"}, "qty": {"type": "integer", "default": 1},
                                                "size_label": {"type": "string"}, "size_code": {"type": "string"},
                                                "notes": {"type": "string"}}, "required": ["index"]}}},
    {"type": "function", "function": {"name": "add_item_by_name",
                                      "description": "Add item by name with optional size and qty (strict match; otherwise suggest only).",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1},
                                                                                      "size_label": {"type": "string"},
                                                                                      "size_code": {"type": "string"},
                                                                                      "notes": {"type": "string"}},
                                                     "required": ["name"]}}},
    {"type": "function", "function": {"name": "add_item_by_code", "description": "Add item by backend code (I_CODE).",
                                      "parameters": {"type": "object", "properties": {"code": {"type": "string"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1},
                                                                                      "size_label": {"type": "string"},
                                                                                      "size_code": {"type": "string"},
                                                                                      "notes": {"type": "string"}},
                                                     "required": ["code"]}}},
    {"type": "function", "function": {"name": "remove_item_by_index",
                                      "description": "Remove item using its index from the last shown items list.",
                                      "parameters": {"type": "object", "properties": {"index": {"type": "integer"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1},
                                                                                      "size_label": {"type": "string"},
                                                                                      "size_code": {"type": "string"}},
                                                     "required": ["index"]}}},
    {"type": "function", "function": {"name": "remove_item_by_name", "description": "Remove item by name.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1},
                                                                                      "size_label": {"type": "string"},
                                                                                      "size_code": {"type": "string"}},
                                                     "required": ["name"]}}},
    {"type": "function", "function": {"name": "remove_from_cart_by_index",
                                      "description": "Remove quantity from the Nth row of the last shown cart.",
                                      "parameters": {"type": "object", "properties": {"index": {"type": "integer"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1}},
                                                     "required": ["index"]}}},
    {"type": "function", "function": {"name": "select_number",
                                      "description": "Handle a bare number by selecting from the last shown list (categories → list_items, items → add_item_by_index, cart → remove_from_cart_by_index).",
                                      "parameters": {"type": "object", "properties": {"index": {"type": "integer"},
                                                                                      "qty": {"type": "integer",
                                                                                              "default": 1}},
                                                     "required": ["index"]}}},
    {"type": "function",
     "function": {"name": "show_cart", "description": "Return current cart lines, total, and currency.",
                  "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "clear_cart", "description": "Clear all items from the cart.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "checkout", "description": "Place the order.",
                                      "parameters": {"type": "object", "properties": {"phone": {"type": "string"},
                                                                                      "address": {"type": "string"}}}}}
]

# ==================== TTS (Voice-Out) - DUAL MODE ====================

# ==================== TTS (Voice-Out) - DUAL MODE ====================
_elevenlabs_client = None


def get_elevenlabs_client():
    global _elevenlabs_client
    if _elevenlabs_client is None and ELEVENLABS_API_KEY:
        _elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _elevenlabs_client




def synthesize_elevenlabs_tts(text: str) -> Optional[bytes]:
    client = get_elevenlabs_client()
    if not client:
        st.warning("ElevenLabs API key not set. JARVIS TTS is disabled.")
        return None

    try:
        # Option A: stream (lowest latency)
        resp = client.text_to_speech.stream(
            voice_id=ELEVENLABS_VOICE_ID,           # e.g. "aCChyB4P5WEomwRsOKRh"
            model_id="eleven_multilingual_v2",      # or "eleven_turbo_v2_5"
            output_format="mp3_22050_32",           # mp3 chunks
            text=text,
            voice_settings=VoiceSettings(
                stability=0.3,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        # Collect chunks -> bytes
        audio_bytes = b"".join([chunk for chunk in resp if chunk])
        return audio_bytes

        # --- OR ---
        # Option B: convert (also returns chunks; good when saving to file)
        # resp = client.text_to_speech.convert(
        #     voice_id=ELEVENLABS_VOICE_ID,
        #     model_id="eleven_turbo_v2_5",
        #     output_format="mp3_22050_32",
        #     text=text,
        #     voice_settings=VoiceSettings(...),
        # )
        # return b"".join([chunk for chunk in resp if chunk])

    except Exception as e:
        st.error(f"ElevenLabs TTS error: {e}")
        return None
def synthesize_openai_tts(text: str) -> Optional[bytes]:
    try:
        # Uses the original OpenAI TTS function for Push to talk mode
        obj = ai().audio.speech.create(model=_tts_model(), voice=_tts_voice(), input=text)
        return obj.read()
    except Exception as e:
        st.warning(f"OpenAI TTS error: {e}")
        return None


def speak_if_enabled(text: str):
    if not text or not ss.get("voice_enabled"): return
    if "spoken_set" not in ss: ss["spoken_set"] = set()

    key = hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()
    if key in ss["spoken_set"]: return

    # Checks the mic_mode and calls the correct TTS function
    audio_bytes = None
    if ss.get("mic_mode") == "JARVIS Mode":
        audio_bytes = synthesize_elevenlabs_tts(text)
    else:
        audio_bytes = synthesize_openai_tts(text)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mpeg", autoplay=True)
        ss["spoken_set"].add(key)
# ==================== Chat History (render past) ====================
def _render_chat():
    for m in ss["messages"]:
        is_user = m["role"] == "user"
        avatar = "👤" if is_user else "🤖"
        avatar_class = "user-avatar" if is_user else "bot-avatar"
        message_class = "user-message" if is_user else "bot-message"

        st.markdown(f"""
        <div class="message-container {message_class}">
            <div class="message-avatar {avatar_class}">{avatar}</div>
            <div class="message-bubble">{m["content"]}</div>
        </div>
        """, unsafe_allow_html=True)




# ==================== Main Layout ====================
left, right = st.columns([7, 3])

with left:
    with st.container(height=750):
        _render_chat()

    if ss.get("flash"):
        speak_if_enabled(ss["flash"])
        ss["flash"] = None

    st.markdown("---")

    # Process a command captured by the wake word listener from the last run
    if ss.get("last_user_utterance"):
        user_text = ss.pop("last_user_utterance")
        if user_text != ss.get("_last_user_text"):
            ss["_last_user_text"] = user_text
            ss["messages"].append({"role": "user", "content": user_text})
            _save_state()
            bot = agent_reply(user_text)
            ss["messages"].append({"role": "assistant", "content": bot})
            _save_state()
            ss["flash"] = bot
        st.rerun()

    if ss.get("mic_mode") == 'JARVIS Mode':
        if ss.get("jarvis_active"):
            if st.button("🔴 Deactivate JARVIS", use_container_width=True):
                ss["jarvis_active"] = False
                st.rerun()
            run_wake_word_loop()
        else:
            if st.button("🔵 Activate JARVIS", use_container_width=True):
                ss["jarvis_active"] = True
                st.rerun()

    else:  # Push to talk Mode
        input_container = st.container()
        with input_container:
            mic_col, chat_col = st.columns([1, 8])
            with chat_col:
                user_text = st.chat_input("Type here… / اكتب هنا…", key=f"chat_input_main_{ss.get('_chat_nonce', 0)}")
            with mic_col:
                audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", format="wav", use_container_width=True,
                                     key=f"mic_rec_btn_{ss.get('_mic_nonce', 0)}")

        if user_text and user_text.strip():
            if user_text != ss.get("_last_user_text"):
                ss["_last_user_text"] = user_text
                ss["messages"].append({"role": "user", "content": user_text})
                _save_state()
                bot = agent_reply(user_text)
                ss["messages"].append({"role": "assistant", "content": bot})
                _save_state()
                ss["flash"] = bot
            ss["_chat_nonce"] = ss.get("_chat_nonce", 0) + 1
            st.rerun()

        if audio and isinstance(audio, dict) and audio.get("bytes") and not ss.get("voice_processing"):
            raw = audio["bytes"]
            h = hashlib.md5(raw).hexdigest()
            if h != ss.get("_last_audio_hash"):
                ss["_last_audio_hash"] = h
                ss["voice_processing"] = True
                with st.spinner("Transcribing..."):
                    text = _transcribe_bytes(raw, lang_hint=ss.get("lang"))
                ss["voice_processing"] = False
                if text and (utter := text.strip()) and utter != ss.get("_last_user_text"):
                    ss["_last_user_text"] = utter
                    ss["messages"].append({"role": "user", "content": utter})
                    _save_state()
                    bot = agent_reply(utter)
                    ss["messages"].append({"role": "assistant", "content": bot})
                    _save_state()
                    ss["flash"] = bot
                elif not text:
                    st.warning("Couldn’t understand the recording.")
            ss["_mic_nonce"] = ss.get("_mic_nonce", 0) + 1
            st.rerun()

with right:
    st.markdown('<div class="cart-panel">', unsafe_allow_html=True)
    st.subheader("🧺 Your Order")

    if not ss["cart"]:
        st.markdown('<p class="empty-cart-text">Your cart is empty.</p>', unsafe_allow_html=True)
    else:
        for it in ss["cart"]:
            size = f" ({it.get('size')})" if it.get("size") and it.get("size").lower() != "default" else ""
            st.markdown(
                f'<div class="cart-line"><span>{it["name"]} &times;{it["qty"]}{size}</span><span>{money(_to_float(it.get("price", 0)) * _to_float(it.get("qty", 0)))}</span></div>',
                unsafe_allow_html=True)
        st.markdown(f'<div class="cart-total"><span>Total</span> <span>{money(cart_total(ss["cart"]))}</span></div>',
                    unsafe_allow_html=True)

        if not ss.get("confirming_clear"):
            if st.button("🗑️ Clear Cart", use_container_width=True):
                ss["confirming_clear"] = True
                st.rerun()
        else:
            st.warning("Are you sure you want to clear the cart?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Yes, clear it", use_container_width=True, type="primary"):
                ss["cart"] = []
                ss["confirming_clear"] = False
                _save_state()
                st.toast("Cart cleared")
                st.rerun()
            if c2.button("❌ No, keep it", use_container_width=True):
                ss["confirming_clear"] = False
                st.rerun()

    st.markdown("---")
    old_addr = ss.get("address", "")
    ss["address"] = st.text_area("Delivery Address (optional)", value=ss.get("address", ""), height=100)
    if ss["address"] != old_addr:
        _save_state()

    if not ss.get("confirming_checkout"):
        if st.button("🛒 Proceed to Checkout", use_container_width=True, type="primary", disabled=not ss["cart"]):
            ss["confirming_checkout"] = True
            st.rerun()

    if ss.get("confirming_checkout"):
        st.info(f"Please confirm your order. Total: **{money(cart_total(ss['cart']))}**")
        c1, c2 = st.columns(2)
        if c1.button("✅ Confirm & Send Order", use_container_width=True):
            with st.spinner("Placing your order..."):
                ss["sending_order"] = True
                r = tool_checkout(ss.get("phone"), ss.get("address", ""))
                ss["sending_order"] = False
                ss["confirming_checkout"] = False
                if r.get("ok"):
                    st.success(f"Order sent! ID: {r.get('order_id')}")
                    ss["messages"].append({"role": "assistant",
                                           "content": f"Order placed successfully! ✅ Your order ID is {r.get('order_id')}."})
                else:
                    st.error(f"Order failed: {r.get('error', 'unknown_error')}")
                _save_state()
                st.rerun()

        if c2.button("↩️ Back to Cart", use_container_width=True):
            ss["confirming_checkout"] = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
