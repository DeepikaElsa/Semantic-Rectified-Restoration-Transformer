import matplotlib.pyplot as plt

from data.datasets import create_dataloader

if __name__ == "__main__":

    loader = create_dataloader(
        root_dir="data/Gopro",
        split="train",
        batch_size=1,
        image_size=256,
        num_workers=0
    )

    batch = next(iter(loader))

    blur = batch["blur"][0].permute(1, 2, 0).numpy()
    sharp = batch["sharp"][0].permute(1, 2, 0).numpy()

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(blur)
    plt.title("Blur")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(sharp)
    plt.title("Sharp")
    plt.axis("off")

    plt.show()