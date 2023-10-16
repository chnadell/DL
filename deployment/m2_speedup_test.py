from pathlib import Path
import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.io.image import read_image
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
sns.set()

if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    print(torch.__version__)

    # test parameters
    n_ims = 20
    batch_sizes = [1, 2, 4, 8, 10, 12, 16, 32, 40, 48, 58, 64]

    cpu_times = []
    mps_times = []
    convert_times = []

    for batch_size in batch_sizes:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        images = [torch.rand((batch_size, 3, 256, 256)) for i in range(n_ims)]

        tic = time.time()
        warmup = model(images[0])
        for image in tqdm(images, desc='CPU images: '):
            result = model(image)
        toc = time.time() - tic
        cpu_times.append(toc / len(images))
        print(f'CPU time per image: {toc / len(images):.3}')

        print("converting images and model to mps")
        model = model.to('mps')
        tic_convert = time.time()
        images = [image.to('mps') for image in images]
        toc_convert = time.time() - tic_convert
        convert_times.append(toc_convert /len(images))


        warmup = model(images[0])
        tic = time.time()
        for image in tqdm(images, desc='MPS images: '):
            result = model(image)
        toc = time.time() - tic
        mps_times.append(toc/len(images))
        print(f'MPS time per image: {toc / len(images):.3}')

    f, ax = plt.subplots()

    ax.scatter(batch_sizes, cpu_times, label='CPU images')
    ax.scatter(batch_sizes, mps_times, label='MPS images')
    ax.scatter(batch_sizes, convert_times, label='Conversion')

    ax.set(
        xlabel='Batch Size',
        ylabel='Time Per Batch (s)',
        title=f'ResNet50, {n_ims} images 256x256, with warmup'
    )
    plt.legend()
    plt.tight_layout()

    df = pd.DataFrame({
        "n_ims": n_ims,
        "Batch Size": batch_sizes,
        "CPU times": cpu_times,
        "MPS times": mps_times,
    })

    out_dir = Path('./deployment/timing_data')
    df.to_csv( out_dir.resolve() / f'mps_cpu_timing_data_{n_ims}.csv', index=False)
    # plt.show()
    f.savefig(out_dir.resolve() / f"mps_cpu_timing_plot_{n_ims}.png", dpi=200)

    print('done')
