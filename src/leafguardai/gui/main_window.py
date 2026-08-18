import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image

from leafguardai.model.infer import format_class_name, load_classes, load_model, predict


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LeafGuard AI")
        self.geometry("720x760")
        ctk.set_default_color_theme("green")
        ctk.set_appearance_mode("dark")

        self.selected_path = None
        self.preview_image = None
        self._model = None
        self._classes = None
        self._device = None

        self.create_widgets()

    def create_widgets(self):
        self.select_button = ctk.CTkButton(
            self,
            text="Selecionar imagem",
            command=self.select_image,
        )
        self.select_button.pack(pady=(24, 8))

        self.analyze_button = ctk.CTkButton(
            self,
            text="Analisar",
            command=self.analyze_image,
            state="disabled",
        )
        self.analyze_button.pack(pady=8)

        self.image_label = ctk.CTkLabel(self, text="Nenhuma imagem selecionada.")
        self.image_label.pack(padx=20, pady=12)

        self.result_label = ctk.CTkLabel(
            self,
            text="Selecione uma folha e clique em Analisar.",
            justify="left",
            wraplength=640,
        )
        self.result_label.pack(padx=24, pady=12)

        self.change_theme_btn = ctk.CTkButton(
            self,
            text="Alternar tema",
            command=self.change_theme,
        )
        self.change_theme_btn.pack(padx=10, pady=16)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Escolher imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.JPG *.JPEG *.bmp"),
            ],
        )

        if not path:
            return

        self.selected_path = path
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 360))
        self.preview_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )
        self.image_label.configure(image=self.preview_image, text="")
        self.analyze_button.configure(state="normal")
        self.result_label.configure(text="Imagem pronta. Clique em Analisar.")

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch

        self.result_label.configure(text="Carregando modelo...")
        self.update_idletasks()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._classes = load_classes()
        self._model = load_model(num_classes=len(self._classes), device=self._device)

    def analyze_image(self):
        if not self.selected_path:
            self.result_label.configure(text="Selecione uma imagem primeiro.")
            return

        try:
            self._ensure_model()
            results = predict(
                self.selected_path,
                top_k=3,
                model=self._model,
                classes=self._classes,
                device=self._device,
            )
        except FileNotFoundError as error:
            self.result_label.configure(text=str(error))
            return
        except Exception as error:
            self.result_label.configure(text=f"Não foi possível analisar a imagem.\n{error}")
            return

        lines = ["Resultado (top-3):"]
        for rank, (label, prob) in enumerate(results, start=1):
            lines.append(f"{rank}. {format_class_name(label)} — {prob * 100:.1f}%")
        self.result_label.configure(text="\n".join(lines))

    def change_theme(self):
        if ctk.get_appearance_mode().lower() == "dark":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
