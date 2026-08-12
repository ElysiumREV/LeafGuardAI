import sys
from torchvision import transforms
from leafguardai.data.loader import get_classes, get_images_by_class
from leafguardai.data.dataset import PlantDataset
from leafguardai.data.split import split_dataset, create_dataloaders

def main():
    print("1. Buscando imagens e classes...", flush=True)
    samples = get_images_by_class()
    classes = get_classes()
    
    print(f"   -> Classes encontradas ({len(classes)}): {classes}", flush=True)
    print(f"   -> Total de imagens encontradas: {len(samples)}", flush=True)

    if len(samples) == 0:
        print("\n❌ Nenhuma imagem foi encontrada na pasta! Verifique a estrutura.", flush=True)
        return

    print("\n2. Criando Dataset...", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PlantDataset(
        samples=samples,
        classes=classes,
        transform=transform
    )

    print("3. Dividindo em conjuntos (Train, Validation, Test)...", flush=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    
    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=32
    )

    print("\n--- RESULTADOS DO DATASET ---", flush=True)
    print(f"Total de imagens : {len(dataset)}", flush=True)
    print(f"Treino (Train)   : {len(train_dataset)}", flush=True)
    print(f"Validação (Val)  : {len(validation_dataset)}", flush=True)
    print(f"Teste (Test)     : {len(test_dataset)}", flush=True)

    print("\n4. Testando o DataDataLoader...", flush=True)
    images, labels = next(iter(train_loader))
    print(f"Batch Images Shape : {images.shape}", flush=True)
    print(f"Batch Labels Shape : {labels.shape}", flush=True)
    print("\n✅ Dataset e DataLoaders funcionando perfeitamente!", flush=True)

if __name__ == "__main__":
    main()