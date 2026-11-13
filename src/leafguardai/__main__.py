from dotenv import find_dotenv, load_dotenv

from leafguardai.gui.main_window import MainWindow


def main():
    # Lê um .env da pasta atual (ou de uma pasta acima), sem sobrescrever variáveis já definidas.
    load_dotenv(find_dotenv(usecwd=True))
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
