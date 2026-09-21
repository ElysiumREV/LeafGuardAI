from leafguardai.data.dataset import PlantDataset
from leafguardai.data.loader import get_classes, get_images_by_class
from leafguardai.data.split import create_dataloaders, split_samples
from leafguardai.vision.transforms import get_eval_transforms, get_train_transforms


def main():
    print("1. Buscando imagens e classes...", flush=True)
    samples = get_images_by_class()
    classes = get_classes()

    print(f"   -> Classes encontradas ({len(classes)}): {classes}", flush=True)
    print(f"   -> Total de imagens encontradas: {len(samples)}", flush=True)

    if len(samples) == 0:
        print(
            "\nNenhuma imagem foi encontrada na pasta! Verifique a estrutura.",
            flush=True,
        )
        return

    print("\n2. Split estratificado e datasets com transforms separados...", flush=True)
    train_samples, validation_samples, test_samples = split_samples(samples)

    train_dataset = PlantDataset(
        samples=train_samples,
        classes=classes,
        transform=get_train_transforms(),
    )
    validation_dataset = PlantDataset(
        samples=validation_samples,
        classes=classes,
        transform=get_eval_transforms(),
    )
    test_dataset = PlantDataset(
        samples=test_samples,
        classes=classes,
        transform=get_eval_transforms(),
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=32,
    )

    print("\n--- RESULTADOS DO DATASET ---", flush=True)
    print(f"Total de imagens : {len(samples)}", flush=True)
    print(f"Treino (Train)   : {len(train_dataset)}", flush=True)
    print(f"Validação (Val)  : {len(validation_dataset)}", flush=True)
    print(f"Teste (Test)     : {len(test_dataset)}", flush=True)

    print("\n3. Testando o DataLoader...", flush=True)
    images, labels = next(iter(train_loader))
    print(f"Batch Images Shape : {images.shape}", flush=True)
    print(f"Batch Labels Shape : {labels.shape}", flush=True)
    print("\nDataset e DataLoaders funcionando.", flush=True)


if __name__ == "__main__":
    main()
