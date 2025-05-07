"""
Streamlit application for chat with OpenAI Assistant API (GPT-4o), supporting text and image input.
"""

import streamlit as st
import logging
import requests

# Configuração da página
st.set_page_config(
    page_title="GPT-4o Assistant Chat",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_API_BASE = "https://api.openai.com/v1"
HEADERS = lambda api_key: {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

def create_thread(api_key):
    resp = requests.post(
        f"{OPENAI_API_BASE}/threads",
        headers=HEADERS(api_key),
        json={}
    )
    if resp.status_code != 200:
        print("Resposta da API:", resp.text)
    resp.raise_for_status()
    return resp.json()["id"]

def upload_file(api_key, image_bytes, image_filename):
    files = {
        "file": (image_filename, image_bytes)
    }
    data = {
        "purpose": "assistants"
    }
    resp = requests.post(
        f"{OPENAI_API_BASE}/files",
        headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "assistants=v2"
        },
        files=files,
        data=data
    )
    if resp.status_code != 200:
        print("Resposta da API (upload_file):", resp.text)
    resp.raise_for_status()
    return resp.json()["id"]

def send_message(thread_id, content, api_key, image_bytes=None, image_filename=None):
    instrucao_idioma = "Responda em português do Brasil."
    content_completo = f"{content}\n{instrucao_idioma}"

    data = {
        "role": "user",
        "content": [{"type": "text", "text": content_completo}]
    }
    if image_bytes and image_filename:
        file_id = upload_file(api_key, image_bytes, image_filename)
        data["content"].append({
            "type": "image_file",
            "image_file": {"file_id": file_id}
        })
    resp = requests.post(
        f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
        headers=HEADERS(api_key),
        json=data
    )
    resp.raise_for_status()
    return resp.json()

def list_messages(thread_id, api_key):
    resp = requests.get(
        f"{OPENAI_API_BASE}/threads/{thread_id}/messages",
        headers=HEADERS(api_key)
    )
    resp.raise_for_status()
    return resp.json()["data"]

def create_run(thread_id, api_key, assistant_id):
    data = {"assistant_id": assistant_id}
    resp = requests.post(
        f"{OPENAI_API_BASE}/threads/{thread_id}/runs",
        headers=HEADERS(api_key),
        json=data
    )
    resp.raise_for_status()
    return resp.json()["id"]

def wait_for_run_completion(thread_id, run_id, api_key):
    import time
    while True:
        resp = requests.get(
            f"{OPENAI_API_BASE}/threads/{thread_id}/runs/{run_id}",
            headers=HEADERS(api_key)
        )
        resp.raise_for_status()
        run = resp.json()
        if run["status"] in ["completed", "failed", "cancelled", "expired"]:
            return run["status"]
        time.sleep(1)

def reconstruct_history(messages):
    """
    Reconstrói o histórico de mensagens no formato [{"role": ..., "content": ...}, ...]
    """
    history = []
    for m in sorted(messages, key=lambda x: x["created_at"]):
        if m["role"] == "user":
            for c in m["content"]:
                if c["type"] == "text":
                    history.append({"role": "user", "content": c["text"]["value"]})
                elif c["type"] == "image_file":
                    history.append({"role": "user", "content": "[Imagem enviada]"})
        elif m["role"] == "assistant":
            for c in m["content"]:
                if c["type"] == "text":
                    history.append({"role": "assistant", "content": c["text"]["value"]})
    return history

def main() -> None:
    st.subheader("🧠 GPT-4o Assistant Chat", divider="gray", anchor=False)

    # Sidebar for OpenAI API key
    with st.sidebar:
        st.markdown("## 🔑 OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            key="openai_api_key"
        )
        st.markdown(
            "Get your key at [platform.openai.com](https://platform.openai.com/account/api-keys)"
        )

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use GPT-4o.")
        st.stop()

    st.info("Usando apenas a API do OpenAI Platform (Assistant/Threads).")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None

    message_container = st.container(height=500, border=True)
    # Exibe TODO o histórico de mensagens
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Digite sua pergunta...", key="chat_input")
        image_file = st.file_uploader("Envie uma imagem (opcional)", type=["png", "jpg", "jpeg"], key="image_input")
        submitted = st.form_submit_button("Enviar")
    if submitted and prompt:
        # Exibe imediatamente a mensagem do usuário
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(prompt)
            if image_file:
                st.image(image_file)
        with message_container.chat_message("assistant", avatar="🤖"):
            with st.spinner(":green[processando...]"):
                if not st.session_state["thread_id"]:
                    st.session_state["thread_id"] = create_thread(api_key)
                image_bytes = image_file.read() if image_file else None
                image_filename = image_file.name if image_file else None
                send_message(
                    st.session_state["thread_id"],
                    prompt,
                    api_key,
                    image_bytes=image_bytes,
                    image_filename=image_filename
                )
                ASSISTANT_ID = "asst_hXtJTncZyNmwB1AwgB2eZylS"
                run_id = create_run(st.session_state["thread_id"], api_key, ASSISTANT_ID)
                wait_for_run_completion(st.session_state["thread_id"], run_id, api_key)
                messages = list_messages(st.session_state["thread_id"], api_key)
                # Atualiza o histórico completo
                st.session_state["messages"] = reconstruct_history(messages)
        # Força recarregamento para mostrar o histórico atualizado
        st.rerun()

if __name__ == "__main__":
    main()
