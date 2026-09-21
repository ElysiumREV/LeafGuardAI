import argparse
from pathlib import Path

from leafguardai.model.infer import format_class_name, predict


def main():
    parser = argparse.ArgumentParser(
        description="Classifica uma imagem de folha com o modelo treinado do LeafGuardAI"
    )
    parser.add_argument("image", type=str, help="Caminho da imagem a ser analisada")
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Quantas classes mais prováveis mostrar (padrão: 3)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Imagem não encontrada: {image_path}")
        return

    results = predict(image_path, top_k=args.top)

    print(f"\nResultado para: {image_path.name}\n")
    for rank, (label, prob) in enumerate(results, start=1):
        print(f"{rank}. {format_class_name(label):<40} {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
