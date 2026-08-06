import customtkinter as ctk

class MainWindow(ctk.CTk):
  def __init__(self):
    super().__init__()
    self.title("LeafGuard AI")
    self.geometry("600x480")
    ctk.set_default_color_theme("green")
    ctk.set_appearance_mode("light")
    self.create_widgets()

  def create_widgets(self):
    self.select_button = ctk.CTkButton(
       self,
       text="Selecionar Imagens",
       command=self.select_image
    )
    self.select_button.pack(pady=20)

    self.analyze_button = ctk.CTkButton(
      self,
      text="Analisar",
      command=self.analyze_image
    )
    self.analyze_button.pack()

    self.result_label = ctk.CTkLabel(
      self,
      text="Nenhuma Imagem analisada."
    )
    self.result_label.pack(padx=20)

    self.change_theme_btn = ctk.CTkButton(
      self,
      text="Change Theme",
      command=self.change_theme
    )
    self.change_theme_btn.pack(padx=10, pady=10)

  def select_image(self):
    print("Selecionar Imagem")

  def analyze_image(self):
    print("Executando AI...")
    self.result_label.configure(
      text="Healthy (90%)"
    )

  def button_callback(self):
    print("button clicked")

  def change_theme(self):
    if ctk.get_appearance_mode().lower() == "dark":
      ctk.set_appearance_mode("light")
    else:
      ctk.set_appearance_mode("dark")
