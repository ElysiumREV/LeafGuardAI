import customtkinter as ctk
from customtkinter import filedialog
from google import genai
from PIL import Image, ImageOps
import os
import queue
import re
import sys
import threading
import tkinter as tk
from pathlib import Path

from leafguardai.model.infer import (
    PLANT_CHOICES,
    confidence_level,
    download_model,
    format_class_name,
    load_classes,
    load_model,
    model_available,
    predict,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

UNKNOWN_PLANT ="Não sei / outra planta"

# Lista de reserva, usada só se não for possível consultar os modelos da API.
FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
MAX_AUTO_CANDIDATES = 6
_EXCLUDED = ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use", "vision", "thinking", "learnlm")
_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?(?:-(.+))?$")


def discover_models(client):
    """Consulta a API e devolve os modelos Gemini Flash disponíveis, do mais novo ao mais antigo."""
    found = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = model.supported_actions or []
        if actions and "generateContent" not in actions:
            continue
        if any(word in name for word in _EXCLUDED):
            continue
        match = _MODEL_RE.match(name)
        if not match:
            continue
        version = tuple(int(x) for x in match.group(1).split("."))
        is_lite = bool(match.group(2))
        suffix = match.group(3) or ""
        # Estáveis (sem sufixo) vêm antes de preview/exp/datados da mesma versão.
        found.append((tuple(-v for v in version), is_lite, suffix != "", name))
    found.sort()
    return [item[3] for item in found]


AUTO_MODEL = "Automático (troca se estiver ocupado)"


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LeafGuard AI")
        self.geometry("720x820")
        ctk.set_default_color_theme("green")
        ctk.set_appearance_mode("dark")
        self._set_app_icon()

        self.selected_path = None
        self.preview_image = None
        self._model = None
        self._classes = None
        self._device = None

        self.minsize(480, 520)
        self._request_id = 0
        self._analyzing = False
        self._wrap_job = None
        self._text_labels = []
        self._ui_queue = queue.Queue()

        self.create_widgets()
        self._text_labels = [self.result_label, self.response_label, self.ai_label]
        self.bind("<Configure>", self._on_resize)
        self._drain_ui_queue()

    def _set_app_icon(self):
        try:
            if sys.platform.startswith("win"):
                self.iconbitmap(str(ASSETS_DIR / "icon.ico"))
            else:
                self._icon_photo = tk.PhotoImage(file=str(ASSETS_DIR / "icon.png"))
                self.iconphoto(True, self._icon_photo)
        except Exception as error:
            # O ícone é só estético: nunca deve impedir o app de abrir.
            print(f"[Ícone] não foi possível carregar: {error}", file=sys.stderr)

    def create_widgets(self):
        self.body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True)
        self.select_button = ctk.CTkButton(
            self.body,
            text="Selecionar imagem",
            command=self.select_image,
        )
        self.select_button.pack(pady=(24, 8))

        self.plant_label = ctk.CTkLabel(self.body, text="Qual é a planta? (opcional)")
        self.plant_label.pack(pady=(8, 0))

        self.plant_menu = ctk.CTkOptionMenu(
            self.body,
            values=[UNKNOWN_PLANT, *sorted(PLANT_CHOICES)],
        )
        self.plant_menu.set(UNKNOWN_PLANT)
        self.plant_menu.pack(pady=(4, 8))

        self.analyze_button = ctk.CTkButton(
            self.body,
            text="Analisar",
            command=self.analyze_image,
            state="disabled",
        )
        self.analyze_button.pack(pady=8)

        self.progress = ctk.CTkProgressBar(self.body, mode="indeterminate", width=300)

        self.image_label = ctk.CTkLabel(self.body, text="Nenhuma imagem selecionada.")
        self.image_label.pack(padx=20, pady=12)

        self.result_label = ctk.CTkLabel(
            self.body,
            text="Selecione uma folha e clique em Analisar.",
            justify="left",
            wraplength=640,
        )
        self.result_label.pack(padx=24, pady=12)

        self.response_label = ctk.CTkLabel(
                    self.body,
                    text="",
                    justify="left",
                    wraplength=640,
                )
        self.response_label.pack(padx=24, pady=12)

        self.ai_label = ctk.CTkLabel(
                            self.body,
                            text="",
                            justify="left",
                            wraplength=640,
                        )
        self.ai_label.pack(padx=24, pady=12)

        self.change_theme_btn = ctk.CTkButton(
            self.body,
            text="Alternar tema",
            command=self.change_theme,
        )
        self.change_theme_btn.pack(padx=10, pady=16)

        key_frame = ctk.CTkFrame(self.body, fg_color="transparent")
        key_frame.pack(padx=20, pady=20)

        self.api_key_entry = ctk.CTkEntry(
            key_frame,
            width=360,
            placeholder_text="Insira a API Key do Gemini (opcional)",
            show="*",
        )
        self.api_key_entry.pack(side="left", padx=(0, 8))
        # Chave vinda do ambiente / .env: já vem preenchida (e mascarada), mas dá para trocar.
        env_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if env_key:
            self.api_key_entry.insert(0, env_key)

        self.toggle_key_btn = ctk.CTkButton(
            key_frame,
            text="Mostrar",
            width=80,
            command=self.toggle_key_visibility,
        )
        self.toggle_key_btn.pack(side="left")

        model_frame = ctk.CTkFrame(self.body, fg_color="transparent")
        model_frame.pack(pady=(0, 16))

        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=[AUTO_MODEL, *FALLBACK_MODELS],
            width=300,
        )
        self.model_menu.set(AUTO_MODEL)
        self.model_menu.pack(side="left", padx=(0, 8))

        self.refresh_models_btn = ctk.CTkButton(
            model_frame,
            text="Atualizar modelos",
            width=130,
            command=self.refresh_models,
        )
        self.refresh_models_btn.pack(side="left")

    def _post(self, callback):
        """Agenda `callback` na thread da interface (seguro de chamar de outras threads)."""
        self._ui_queue.put(callback)

    def _drain_ui_queue(self):
        try:
            while True:
                self._ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(30, self._drain_ui_queue)

    def _on_resize(self, event):
        if event.widget is not self:
            return
        if self._wrap_job is not None:
            self.after_cancel(self._wrap_job)
        self._wrap_job = self.after(80, lambda: self._apply_wrap(event.width))

    def _apply_wrap(self, width):
        self._wrap_job = None
        wrap = max(280, width - 90)
        for label in self._text_labels:
            label.configure(wraplength=wrap)

    def _progress(self, on):
        if on:
            self.progress.pack(after=self.analyze_button, pady=(0, 8))
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def refresh_models(self):
        api_key = "".join(self.api_key_entry.get().split())
        if not api_key:
            self.response_label.configure(text="Insira a chave da API para listar os modelos.")
            return
        self.refresh_models_btn.configure(state="disabled", text="Consultando...")
        threading.Thread(target=self._refresh_models_worker, args=(api_key,), daemon=True).start()

    def _refresh_models_worker(self, api_key):
        try:
            models = discover_models(genai.Client(api_key=api_key))
            error = None
        except Exception as exc:
            models, error = [], exc

        def apply():
            self.refresh_models_btn.configure(state="normal", text="Atualizar modelos")
            if error is not None:
                self.response_label.configure(text=self._describe_ai_error(error))
            elif not models:
                self.response_label.configure(text="Nenhum modelo Gemini Flash disponível para esta chave.")
            else:
                current = self.model_menu.get()
                self.model_menu.configure(values=[AUTO_MODEL, *models])
                self.model_menu.set(current if current in models else AUTO_MODEL)
                self.response_label.configure(text=f"{len(models)} modelos disponíveis. Mais novo: {models[0]}.")

        self._post(apply)

    def toggle_key_visibility(self):
        if self.api_key_entry.cget("show") == "*":
            self.api_key_entry.configure(show="")
            self.toggle_key_btn.configure(text="Ocultar")
        else:
            self.api_key_entry.configure(show="*")
            self.toggle_key_btn.configure(text="Mostrar")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Escolher imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.JPG *.JPEG *.bmp"),
            ],
        )

        if not path:
            return

        # Descarta qualquer análise/consulta à IA em andamento da imagem anterior.
        self._request_id += 1
        self._progress(False)
        self.response_label.configure(text="")
        self.ai_label.configure(text="")

        self.selected_path = path
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as error:
            self.selected_path = None
            self.analyze_button.configure(state="disabled")
            self.result_label.configure(text=f"Não foi possível abrir a imagem.\n{error}")
            return
        image.thumbnail((360, 360))
        self.preview_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )
        self.image_label.configure(image=self.preview_image, text="")
        self.analyze_button.configure(state="normal")
        self.result_label.configure(text="Imagem pronta. Clique em Analisar.")

    def _ensure_model(self, status):
        if self._model is not None:
            return

        import torch

        if not model_available():
            status("Baixando o modelo (só na primeira vez)...")
            last = [None]

            def on_progress(done, total):
                if total:
                    key, detail = int(done * 100 / total), f"{int(done * 100 / total)}%"
                else:
                    key, detail = done // 2_000_000, f"{done / 1e6:.0f} MB"
                if key != last[0]:  # só atualiza a tela quando o valor exibido muda
                    last[0] = key
                    status(f"Baixando o modelo (só na primeira vez)... {detail}")

            download_model(progress=on_progress)

        status("Carregando modelo...")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._classes = load_classes()
        self._model = load_model(num_classes=len(self._classes), device=self._device)

    def analyze_image(self):
        if self._analyzing:
            return
        if not self.selected_path:
            self.result_label.configure(text="Selecione uma imagem primeiro.")
            return

        # Lê os widgets aqui (thread da interface); a thread de trabalho não toca neles.
        plant_name = self.plant_menu.get()
        plant = PLANT_CHOICES.get(plant_name)
        api_key = "".join(self.api_key_entry.get().split())
        model_choice = self.model_menu.get()

        self._request_id += 1
        rid = self._request_id
        self._analyzing = True
        self.analyze_button.configure(state="disabled")
        self.select_button.configure(state="disabled")
        self.response_label.configure(text="")
        self.ai_label.configure(text="")
        self.result_label.configure(text="Analisando...")
        self._progress(True)

        threading.Thread(
            target=self._analyze_worker,
            args=(rid, self.selected_path, plant_name, plant, api_key, model_choice),
            daemon=True,
        ).start()

    def _analyze_worker(self, rid, path, plant_name, plant, api_key, model_choice):
        def status(text):
            self._post(lambda: self.result_label.configure(text=text))

        try:
            self._ensure_model(status)
            results = predict(
                path,
                top_k=3,
                model=self._model,
                classes=self._classes,
                device=self._device,
                plant=plant,
            )
            error = None
        except Exception as exc:
            results, error = None, exc

        self._post(
            lambda: self._on_analysis_done(rid, results, error, plant_name, plant, api_key, model_choice)
        )

    def _on_analysis_done(self, rid, results, error, plant_name, plant, api_key, model_choice):
        self._analyzing = False
        self.select_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        if rid != self._request_id:
            return

        if error is not None:
            self._progress(False)
            if isinstance(error, FileNotFoundError):
                self.result_label.configure(text=str(error))
            else:
                self.result_label.configure(text=f"Não foi possível analisar a imagem.\n{error}")
            return

        if not results:
            self._progress(False)
            self.result_label.configure(text="Nenhum resultado foi encontrado.")
            return

        top_label, top_prob = results[0]
        level = confidence_level(top_prob, plant_known=plant is not None)

        lines = [f"Resultado (considerando apenas: {plant_name}):" if plant else "Resultado (top-3):"]
        for rank, (label, prob) in enumerate(results, start=1):
            lines.append(f"{rank}. {format_class_name(label)} — {prob * 100:.1f}%")
        if level == "media":
            lines.append("\n⚠ Confiança moderada: o resultado pode estar errado. Confira também as outras opções.")
        elif level == "baixa":
            lines.append("\n⚠ Resultado incerto: o modelo não tem segurança sobre esta imagem.")
        self.result_label.configure(text="\n".join(lines))

        if level == "baixa":
            # Abaixo do limite, o top-1 erra mais do que acerta: não recomendar tratamento.
            self._progress(False)
            self.response_label.configure(
                text="Para um resultado mais confiável, tente outra foto: uma única folha, "
                "bem iluminada e em foco, ocupando a maior parte da imagem."
            )
            return

        if top_label.lower().endswith("___healthy"):
            self._progress(False)
            message = "A folha foi classificada como saudável. Nenhuma doença foi detectada."
            if level == "media":
                message = "A folha parece saudável, mas a confiança é moderada. Se houver manchas ou sintomas, tente outra foto."
            self.response_label.configure(text=message)
            return

        if not api_key:
            self._progress(False)
            self.response_label.configure(text="Insira sua chave da API Gemini para ver orientações de tratamento.")
            return

        top1 = f"{format_class_name(top_label)} — {top_prob * 100:.1f}%"
        prompt = (f"Explique o tratamento apenas da doença identificada como Top 1: {top1}. "
            "Escreva em português e siga exatamente esta estrutura: "
            "Tratamento: explique os principais métodos de tratamento de forma clara, com 2 a 3 frases."
            "Cuidados: informe 2 ou 3 cuidados importantes, de forma resumida."
            "Prevenção: explique 1 ou 2 medidas para evitar o reaparecimento ou disseminação da doença."
            "Não adicione introduções, saudações, conclusões ou outras seções. Não omita nenhuma das três seções. Não mostre colchetes ou instruções do prompt. Mantenha o resultado informativo, mas sem textos longos ou explicações excessivamente técnicas.")

        # A consulta à IA roda em outra thread; a barra de progresso continua até ela terminar.
        self.response_label.configure(text="Consultando IA para orientações... Aguarde.")
        threading.Thread(
            target=self._fetch_ai_guidance,
            args=(prompt, api_key, model_choice, rid),
            daemon=True,
        ).start()

    def _show_ai(self, rid, response, note):
        if rid != self._request_id:
            return  # o usuário já trocou de imagem ou analisou de novo
        self._progress(False)
        self.response_label.configure(text=response)
        self.ai_label.configure(text=note)

    @staticmethod
    def _is_retryable(error):
        """Erros em que vale tentar outro modelo (alto uso, cota, indisponível)."""
        code = getattr(error, "code", None) or getattr(error, "status_code", None)
        low = str(error).lower()
        return code in (404, 429, 500, 503, 504) or any(
            w in low
            for w in ("overloaded", "unavailable", "high demand", "resource_exhausted", "quota", "not found")
        )

    def _fetch_ai_guidance(self, prompt, api_key, selected_model, rid):
        client = genai.Client(api_key=api_key)

        if selected_model == AUTO_MODEL:
            try:
                candidates = discover_models(client)[:MAX_AUTO_CANDIDATES]
            except Exception as error:
                print(f"[Gemini] listagem de modelos falhou: {error}", file=sys.stderr)
                candidates = []
            candidates = candidates or list(FALLBACK_MODELS)
        else:
            candidates = [selected_model]
        failed = []
        last_error = None

        for model in candidates:
            try:
                interaction = client.interactions.create(model=model, input=prompt)
            except Exception as error:
                last_error = error
                print(f"[Gemini] {model}: {type(error).__name__}: {error}", file=sys.stderr)
                if self._is_retryable(error):
                    failed.append(model)
                    continue
                break

            text = interaction.output_text
            note = f"Modelo: {model}"
            if failed:
                note += f" (indisponíveis agora: {', '.join(failed)})"
            note += (
                "\nEsta resposta foi gerada por IA. Para obter orientações mais confiáveis "
                "e adequadas ao caso, recomenda-se consultar um profissional especializado."
            )
            self._post(lambda: self._show_ai(rid, text, note))
            return

        message = self._describe_ai_error(last_error)
        if failed:
            message = f"Modelos tentados sem sucesso: {', '.join(failed)}.\n\n{message}"
        self._post(lambda: self._show_ai(rid, message, ""))

    @staticmethod
    def _describe_ai_error(error):
        raw = str(error)
        low = raw.lower()
        code = getattr(error, "code", None) or getattr(error, "status_code", None)

        if code in (400, 401, 403) or "api key" in low or "api_key" in low or "permission" in low:
            reason = "Chave da API inválida ou sem permissão. Confira se copiou a chave inteira."
        elif code == 404 or "not found" in low:
            reason = "Modelo não encontrado para a sua chave."
        elif code == 429 or "quota" in low or "resource_exhausted" in low:
            reason = "Limite de uso (cota) da API atingido. Tente mais tarde."
        elif any(w in low for w in ("connect", "timeout", "resolve", "network")):
            reason = "Falha de conexão. Verifique sua internet."
        else:
            reason = "Não foi possível obter as orientações da IA."

        print(f"[Gemini] {type(error).__name__}: {raw}", file=sys.stderr)
        return f"{reason}\n\nDetalhe: {type(error).__name__}: {raw[:300]}"

    def change_theme(self):
        if ctk.get_appearance_mode().lower() == "dark":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
