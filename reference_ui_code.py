import streamlit as st
import os
import uuid  # הוספנו לצורך יצירת מזהה ייחודי לכל שיחה
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from streamlit_mic_recorder import mic_recorder
from graph import app

# טעינת הגדרות
load_dotenv("env")

st.set_page_config(page_title="המספרה של נהוראי", page_icon="💈", layout="centered")

# --- עיצוב וואטסאפ פרימיום ---
st.markdown("""
    <style>
    .stApp {
        background-color: #e5ddd5;
        background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
        background-repeat: repeat;
        background-attachment: fixed;
    }
    .main { direction: rtl; }

    /* בועות צ'אט */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 5px;
        max-width: 85%;
        border: none;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff !important;
        align-self: flex-start;
        border-top-right-radius: 0px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    [data-testid="stChatMessageUser"] {
        background-color: #dcf8c6 !important;
        align-self: flex-end;
        border-top-left-radius: 0px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        margin-right: auto;
    }
    [data-testid="stChatMessageContent"] { text-align: right; direction: rtl; }

    /* עיצוב התמונות בתוך הבועה */
    img { 
        border-radius: 8px; 
        border: 1px solid #ddd; 
        margin-top: 5px;
        margin-bottom: 5px; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- ניהול זיכרון ומצב (State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "config" not in st.session_state:
    # יצירת thread_id ייחודי בכל פעם שהאפליקציה עולה מחדש
    unique_id = str(uuid.uuid4())
    st.session_state.config = {"configurable": {"thread_id": unique_id}}

if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None

# --- סרגל צד (Sidebar) לניהול בדיקות ---
with st.sidebar:
    st.header("🛠️ ניהול בדיקות")
    if st.button("🗑️ נקה שיחה (התחל מחדש)"):
        # איפוס ההודעות ב-UI
        st.session_state.messages = []
        # יצירת מזהה ת'רד חדש לגמרי בגרף (הסוכן ישכח הכל)
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st.session_state.last_processed_audio_id = None
        st.rerun()

    st.divider()
    st.caption(f"**Session ID:** `{st.session_state.config['configurable']['thread_id']}`")

st.title("💈 המספרה של נהוראי")

# הצגת היסטוריה
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "✂️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if "image" in msg and msg["image"]:
                if "manager" in msg["image"]:
                    st.image(msg["image"], use_container_width=True)
                else:
                    st.image(msg["image"], width=400)
            if msg["content"]:
                st.write(msg["content"])


# --- פונקציית עיבוד ---
def process_and_display(user_input, mode="text"):
    display_text = "🎤 [הודעה קולית]" if mode == "voice" else user_input

    # 1. הצגה מיידית למשתמש ב-UI
    st.session_state.messages.append({"role": "user", "content": display_text})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.write(display_text)

    # 2. הרצת הגרף
    with chat_container:
        with st.chat_message("assistant", avatar="✂️"):
            with st.spinner("חושב..."):
                inputs = {
                    "messages": [HumanMessage(content=user_input)],
                    "input_mode": mode
                }

                try:
                    # שימוש ב-config הדינמי מה-session_state
                    result = app.invoke(inputs, config=st.session_state.config)
                    answer = result.get("final_response", "")
                    photo = result.get("photo_path")

                    new_msg = {"role": "assistant", "content": answer, "image": None}

                    if photo and os.path.exists(photo):
                        if "manager" in photo:
                            st.image(photo, use_container_width=True)
                        else:
                            st.image(photo, width=400)
                        new_msg["image"] = photo

                    if answer:
                        st.write(answer)

                    st.session_state.messages.append(new_msg)

                except Exception as e:
                    st.error(f"אופס, קרתה תקלה: {str(e)}")

    st.rerun()


# --- אזור הקלט ---
input_col, mic_col = st.columns([0.88, 0.12])
with mic_col:
    audio = mic_recorder(start_prompt="🎤", stop_prompt="✅", key='recorder')
with input_col:
    if prompt := st.chat_input("הודעה"):
        process_and_display(prompt, mode="text")

# טיפול באודיו
if audio:
    audio_id = hash(audio['bytes'])
    if st.session_state.last_processed_audio_id != audio_id:
        st.session_state.last_processed_audio_id = audio_id
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio['bytes'])
        process_and_display(audio_path, mode="voice")