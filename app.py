import os
import torch
import gradio as gr
import requests
import time
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

# 用來產生地圖
import folium
from geopy.geocoders import Nominatim

# 用來產生 PDF
from fpdf import FPDF
import io  # 用於處理記憶體檔案
from io import BytesIO # 用於圖片轉為記憶體檔案
from base64 import b64encode  # 用來將 PDF 轉成 base64 字串以便 HTML 下載 

import warnings

# 直接從環境變數取得金鑰（Hugging Face 會自動注入 Secret）
api_key = os.environ.get("GROQ_API_KEY")
model = "meta-llama/llama-4-scout-17b-16e-instruct"
api_url = "https://api.groq.com/openai/v1/chat/completions"

# 檢查金鑰是否存在
if not api_key:
    raise ValueError("❌ 找不到 Groq API 金鑰，請先設定環境變數！")

print(f"✅ 使用模型：GROQ ({model})")
print(f"✅ API Endpoint：{api_url}")

# ✅ Groq API 金鑰驗證
def check_groq_api_key(api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Groq API 金鑰驗證成功！可以正常使用！")
            print("✅ 測試回應:", response.json()["choices"][0]["message"]["content"])
            return True
        else:
            print(f"❌ Groq API 錯誤！狀態碼: {response.status_code}")
            print("錯誤訊息:", response.text)
            return False
    except Exception as e:
        print(f"❌ 發生例外錯誤: {e}")
        return False

# 金鑰驗證
check_groq_api_key(api_key)

# ✅ LLM (Groq API)回應
def llm_reply(prompt, chat_history=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # 支援多輪聊天，傳遞過去歷史訊息
    messages = chat_history if chat_history else []
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300
    }
    try:
        response = requests.post(api_url, headers=headers, json=body)
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"✅ LLM 回應: {reply}")
            messages.append({"role": "assistant", "content": reply})
            return reply, messages
        else:
            print(f"❌ Groq API 錯誤: {response.status_code} - {response.text}")
            return f"❌ Groq API 錯誤: {response.status_code}", messages
    except Exception as e:
        print(f"❌ 發生例外錯誤: {e}")
        return f"❌ 發生例外錯誤: {e}", messages


# 📌 Diffusion 模型初始化
pipe = None

def load_diffusion_model():
    global pipe
    if pipe is None:
        print("⏳ 正在加載 Diffusion 模型...")
        model_id = "stabilityai/stable-diffusion-2-1"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ 模型加載完成！")
    else:
        print("✅ 模型已經加載，直接使用！")
    return pipe

STYLE_PRESETS = {
    "手繪風": "hand-drawn style, illustration, warm color palette, soft warm tones, high detail",
    "日系可愛": "anime style, cute, Japanese art, pastel colors, soft light",
    "寫實風": "realistic style, high detail, cinematic lighting, 4k resolution",
    "像素風": "pixel art, 8-bit style, retro game aesthetics, bright colors",
    "水彩風": "watercolor painting, soft edges, delicate brush strokes, muted colors"
}


# 📌 Diffusion 圖像生成
DEFAULT_STYLE_KEY = "手繪風"

def generate_cover_image(prompt, style_choice, default_key=DEFAULT_STYLE_KEY):
    STYLE_PRESETS = {
        "手繪風": "hand-drawn style, illustration, warm color palette, soft warm tones, high detail",
        "日系可愛": "anime style, cute, Japanese art, pastel colors, soft light",
        "寫實風": "realistic style, high detail, cinematic lighting, 4k resolution",
        "像素風": "pixel art, 8-bit style, retro game aesthetics, bright colors",
        "水彩風": "watercolor painting, soft edges, delicate brush strokes, muted colors"
    }
    try:
        load_diffusion_model()
        if style_choice in STYLE_PRESETS:
            style = STYLE_PRESETS[style_choice]
            print(f"🎨 使用者選擇風格：「{style_choice}」")
        else:
            style = STYLE_PRESETS[default_key]
            print(f"🎨 使用者未選擇風格或選擇無效，已套用預設風格：「{default_key}」")
        full_prompt = f"travel guide cover, {prompt}, {style}"
        print(f"🎨 開始生成圖片：{full_prompt}")
        start_time = time.time()
        result = pipe(full_prompt, num_inference_steps=30, guidance_scale=7.5)
        image = result.images[0]
        duration = time.time() - start_time
        print(f"✅ 圖片生成完成，用時 {duration:.1f} 秒")
        return image
    except Exception as e:
        print(f"❌ 發生例外錯誤：{e}")
        return Image.new("RGB", (512, 512), color="gray")

# 📌 AI Agents - Reflection & Planning Prompt
def agent_plan_route(location, preference, budget, days, group, transport, season="自動推斷"):
    # Reflection + Planning Prompt
    prompt = (
        f"你是資深旅遊規劃AI，具備專業知識與即時判斷能力。\n"
        f"【旅遊資訊】\n"
        f"地點：{location}\n"
        f"偏好：{preference}\n"
        f"預算：{budget}\n"
        f"天數：{days}\n"
        f"人數/身份：{group}\n"
        f"交通方式：{transport}\n"
        f"時節/季節：{season}\n"
        f"請針對上述需求，進行：\n"
        f"1. 完整行程自動規劃，並列出每日路線\n"
        f"2. Reflection：檢查有無「明顯不合理」安排（如冬天安排賞櫻、交通中斷等），自動修正，並說明調整理由\n"
        f"3. 行程包含天氣/景點/交通/餐飲等推薦，禁止出現無法實現的內容\n"
        f"4. 以條列式輸出"
    )
    result, _ = llm_reply(prompt)
    return result

# 📌 顯示地圖
def generate_map_html(location_name):
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(location_name)
    if location:
        m = folium.Map(location=[location.latitude, location.longitude], zoom_start=13)
        folium.Marker([location.latitude, location.longitude], popup=f"{location_name}").add_to(m)
        return m._repr_html_()
    else:
        return "⚠️ 找不到地點，請確認輸入是否正確"

# 📌#PDF
#加圖片版(目前用這個)
# -*- coding: utf-8 -*-
# 下載支援中文字型的 Noto Sans CJK 字型檔（Regular 與 Bold）
# !wget -O NotoSansCJKtc-Regular.ttf "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
# !wget -O NotoSansCJKtc-Bold.otf "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Bold.otf"

# 定義產生 PDF 並內嵌圖片與文字內容的函式
# def generate_and_display_pdf(image, text="hello world", filename="fpdf2-demo.pdf", font_size=14, width=800, height=400, font_path="/content/NotoSansCJKtc-Regular.ttf"):
def generate_and_display_pdf(
    image, text="hello world", filename="fpdf2-demo.pdf", font_size=14, width=800, height=400,
    font_path="NotoSansCJKtc-Regular.otf", bold_font_path="NotoSansCJKtc-Bold.otf"
):
    # 顯示 Python 的棄用警告（有助於除錯）
    warnings.simplefilter('default', DeprecationWarning)

    # 建立 PDF 文件物件
    pdf = FPDF()
    pdf.add_page()  # 加入一頁

    # 註冊並使用中文字型
    pdf.add_font('NotoSansCJKtc', '', font_path, uni=True)  # Regular
    pdf.set_font('NotoSansCJKtc', size=font_size)  # 預設字型
    # pdf.add_font('NotoSansCJKtc', 'B', 'NotoSansCJKtc-Bold.otf')  # Bold
    pdf.add_font('NotoSansCJKtc', 'B', bold_font_path, uni=True)
    pdf.set_font('NotoSansCJKtc', style='B', size=font_size)  # 設定為粗體

    # ===== 圖片頁首處理 =====
    # image_buffer = io.BytesIO()  # 建立記憶體檔案物件
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")  # 將 PIL 圖片存為 PNG 格式
    image_buffer.seek(0)  # 將檔案指標移至開頭
    pdf.image(image_buffer, x=10, y=20, w=pdf.w - 20)  # 將圖片插入 PDF 頁面
    pdf.ln(100)  # 留空行距避免文字與圖片重疊

    # ===== 處理後續文字內容 =====
    pdf.add_page()  # 加入一頁
    pdf.set_font('NotoSansCJKtc', '', font_size)  # 恢復為 regular 字型
    safe_width = pdf.w - 2 * pdf.l_margin  # 可用文字寬度（扣掉左右邊界）

    # 將輸入文字逐行處理
    for line in text.split("\n"):
        line = line.rstrip()  # 移除右邊空白

        if not line:  # 空行：換段落
            pdf.ln(font_size//2 + 2)

        elif line.startswith('### ') or line.startswith('#### '):  # 標題格式
            pdf.set_font('NotoSansCJKtc', size=font_size+4, style='B')
            pdf.multi_cell(safe_width, font_size+10, line[4:])  # 切除 ###
            pdf.set_font('NotoSansCJKtc', size=font_size)

        elif line.startswith('**') and line.endswith('**'):  # 粗體行
            pdf.set_font('NotoSansCJKtc', size=font_size+2, style='B')
            pdf.multi_cell(safe_width, font_size+6, line.replace('**',''))
            pdf.set_font('NotoSansCJKtc', size=font_size)

        elif line.startswith('*'):  # 列表項目
            pdf.set_x(pdf.l_margin + 10)  # 稍微縮排
            pdf.multi_cell(safe_width-10, font_size, line[1:].strip())

        else:  # 一般段落
            if '**' in line:
                segments = []
                while '**' in line:
                    pre, rest = line.split('**', 1)
                    if '**' not in rest:
                        segments.append((pre + '**' + rest, False))  # 沒有配對，當作普通字
                        break
                    bold, line = rest.split('**', 1)
                    segments.append((pre, False))
                    segments.append((bold, True))
                segments.append((line, False))

                line_height = font_size + 1
                max_width = safe_width
                curr_width = 0
                for seg_text, is_bold in segments:
                    seg_text = seg_text.strip()
                    if not seg_text:
                        continue
                    pdf.set_font('NotoSansCJKtc', style='B' if is_bold else '', size=font_size)
                    # 中文計算寬度
                    text_width = pdf.get_string_width(seg_text)
                    # 斷行（超出就跳行）
                    i = 0
                    while i < len(seg_text):
                      char = seg_text[i]
                      char_width = pdf.get_string_width(char)
                      if curr_width + char_width > max_width:
                        pdf.ln(line_height)                        
                        pdf.set_x(pdf.l_margin + 5)
                        curr_width = 0
                      pdf.cell(char_width, line_height, char, ln=0)
                      curr_width += char_width
                      i += 1
                pdf.ln(line_height)
            else:
                pdf.set_font('NotoSansCJKtc', style='', size=font_size)
                pdf.set_x(pdf.l_margin + 5)  # 稍微縮排
                pdf.multi_cell(safe_width, font_size + 2, line, align='L')

    # 將 PDF 轉為 byte 並編碼成 base64 字串
    # pdf_bytes = pdf.output()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    base64_pdf = b64encode(pdf_bytes).decode("utf-8")

    # 回傳 HTML 的下載連結
    html_download = f'<a download="{filename}" href="data:application/pdf;base64,{base64_pdf}">下載 PDF</a>'
    return html_download

# 📌設定 Gradio 介面
with gr.Blocks() as demo:
    gr.Markdown("## 🌏 AI旅遊嚮導：結合圖文生成的個人化旅遊推薦系統")

    with gr.Tab("個人化旅遊推薦"):
        with gr.Row():
            location = gr.Textbox(label="旅遊地點/主題", placeholder="東京鐵塔、京都櫻花")
            preference = gr.Textbox(label="旅遊偏好", placeholder="自然景觀/人文藝術/美食/購物")
            budget = gr.Textbox(label="預算（可選）", placeholder="例如：2萬元以內")
            days = gr.Textbox(label="天數（可選）", placeholder="例如：5天4夜")
            group = gr.Textbox(label="人數/身份（可選）", placeholder="2人情侶/全家出遊")
            # transport = gr.Textbox(label="交通方式（可選）", placeholder="自駕/大眾運輸")

            transport = gr.Dropdown(
                choices=["自駕", "大眾運輸", "包車", "步行", "自行車", "不限/其他"],
                label="交通方式（可選）",
                value="自駕",  # 預設值，可依需求調整
                allow_custom_value=True  # 允許自訂輸入（如需完全限制選項可設為 False）
            )

            style_choice = gr.Radio(
                choices=list(STYLE_PRESETS.keys()),
                value=DEFAULT_STYLE_KEY,
                label="圖像風格（請選擇一項）"
            )          


        with gr.Row():
            show_options = gr.CheckboxGroup(
                choices=["顯示圖像", "顯示地圖", "下載 PDF"],
                label="顯示選項",
                value=["顯示圖像"],  # 預設勾選，可自行調整
                type="value"
            )
            submit = gr.Button("生成")

        llm_output = gr.Textbox(label="旅遊建議（RAG+LLM推薦景點）", lines=5)
        image_output = gr.Image(label="專屬旅遊場景圖片", visible=True)
        agent_output = gr.Textbox(label="AI Agents 行程規劃 (Reflection & Planning)", lines=10)
        map_display = gr.HTML(label="地圖顯示", visible=True)
        pdf_download = gr.HTML(label="下載 PDF", visible=True)
        error_output = gr.Textbox(label="錯誤訊息", lines=3)

        def ai_travel_assistant_all(location, preference, budget, days, group, transport, style_choice, show_image, show_map, show_pdf, pdf_name_input="cd.pdf"):
            try:
                user_prompt = (
                    f"請根據以下資訊規劃個人化旅遊建議，包含路線/景點/活動。\n"
                    f"地點：{location}\n"
                    f"旅遊偏好：{preference}\n"
                    f"預算：{budget}\n"
                    f"天數：{days}\n"
                    f"人數或身份：{group}\n"
                    f"交通方式：{transport}\n"
                    f"請條列化建議內容。"
                )
                llm_result, _ = llm_reply(user_prompt)
                image_result = generate_cover_image(location, style_choice) if show_image else None
                agent_result = agent_plan_route(location, preference, budget, days, group, transport)
                map_html = generate_map_html(location) if show_map else ""
                pdf_text = llm_result + agent_result
                PP = generate_and_display_pdf(image=image_result, text=pdf_text, filename="AI_Travel_Plan.pdf", font_size=14, width=800, height=400) if show_pdf else ""
                return llm_result, image_result, agent_result, map_html, PP, ""
            except Exception as e:
                return "發生錯誤，請稍後再試", None, "", "", None, str(e)

        submit.click(
            fn=ai_travel_assistant_all,
            inputs=[location, preference, budget, days, group, transport, style_choice, show_options],
            outputs=[llm_output, image_output, agent_output, map_display, pdf_download, error_output]
        )


        # with gr.Row():
        #     submit = gr.Button("生成旅遊建議、圖像、地圖與 PDF")
        # llm_output = gr.Textbox(label="旅遊建議（RAG+LLM推薦景點）", lines=5)
        # image_output = gr.Image(label="專屬旅遊場景圖片")
        # agent_output = gr.Textbox(label="AI Agents 行程規劃 (Reflection & Planning)", lines=10)
        # map_display = gr.HTML(label="地圖顯示") # ⬅ 新增地圖顯示欄位
        # pdf_download = gr.HTML(label="下載 PDF")
        # error_output = gr.Textbox(label="錯誤訊息", lines=3)

        # def ai_travel_assistant_all(location, preference, budget, days, group, transport, style_choice, pdf_name_input="cd.pdf"):
        #     try:
        #         user_prompt = (
        #         f"請根據以下資訊規劃個人化旅遊建議，包含路線/景點/活動。\n"
        #         f"地點：{location}\n"
        #         f"旅遊偏好：{preference}\n"
        #         f"預算：{budget}\n"
        #         f"天數：{days}\n"
        #         f"人數或身份：{group}\n"
        #         f"交通方式：{transport}\n"
        #         f"請條列化建議內容。"
        #         )
        #         # 1) LLM 文字建議
        #         llm_result, _ = llm_reply(user_prompt)

        #         # 2) 生成封面圖
        #         image_result = generate_cover_image(location, style_choice)

        #         # 3) AI Agents (Reflection + Planning)
        #         agent_result = agent_plan_route(location, preference, budget, days, group, transport)

        #         # 4) 用 generate_map_html 函數生成地圖
        #         map_html = generate_map_html(location)  # ⬅ 加入地圖

        #         # 5) PDF
        #         pdf_text = llm_result + agent_result
        #         PP = generate_and_display_pdf(image=image_result ,text=pdf_text, filename="AI_Travel_Plan.pdf", font_size=14, width=800, height=400)

        #         return llm_result, image_result, agent_result, map_html, PP, ""
        #     except Exception as e:
        #         # 發生錯誤時回傳錯誤訊息，其它欄位設為空或 None
        #         return "發生錯誤，請稍後再試", None, "", "", None, str(e)

        # submit.click(
        #     fn=ai_travel_assistant_all,
        #     inputs=[location, preference, budget, days, group, transport, style_choice],
        #     outputs=[llm_output, image_output, agent_output, map_display, pdf_download, error_output]
        # )

    with gr.Tab("互動式旅遊聊天機器人"):
        chatbox = gr.Chatbot(label="旅遊小助手：自由提問、推薦、查詢、規劃皆可")
        chat_input = gr.Textbox(label="請輸入你的問題", placeholder="請問大阪春天有什麼活動？")
        chat_submit = gr.Button("發送")
        chat_state = gr.State([])  # List of messages

        def chat_ai(user_msg, chat_history):
            reply, new_history = llm_reply(user_msg, chat_history)
            chat_history.append((user_msg, reply))
            return chat_history, new_history

        chat_submit.click(
            fn=chat_ai,
            inputs=[chat_input, chat_state],
            outputs=[chatbox, chat_state]
        )

    demo.launch()
