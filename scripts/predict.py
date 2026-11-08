import argparse
from pathlib import Path

from leafguardai.model.infer import PLANT_CHOICES, confidence_level, format_class_name, predict


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
    parser.add_argument(
        "--planta",
        choices=sorted(PLANT_CHOICES),
        help="Planta da foto, se conhecida: considera só as classes dela",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Imagem não encontrada: {image_path}")
        return

    plant = PLANT_CHOICES.get(args.planta)
    results = predict(image_path, top_k=args.top, plant=plant)

    print(f"\nResultado para: {image_path.name}\n")
    for rank, (label, prob) in enumerate(results, start=1):
        print(f"{rank}. {format_class_name(label):<40} {prob * 100:.2f}%")

    level = confidence_level(results[0][1], plant_known=plant is not None) if results else "baixa"
    if level == "media":
        print("\nAviso: confiança moderada; o resultado pode estar errado. Confira as outras opções.")
    elif level == "baixa":
        print("\nAviso: resultado incerto. Tente outra foto: uma única folha, bem iluminada e em foco.")


if __name__ == "__main__":
    main()
