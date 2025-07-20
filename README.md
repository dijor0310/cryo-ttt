# Denoising test-time training for cryo-ET membrane segmentation 🧊 🦠

This repository contains the implementation of denoising test-time training for membrane segmentation in cryo-electron tomography (cryo-ET). The goal is to improve segmentation performance on noisy tomograms by adapting the model at test time.

## Installation ⚙️ 

This project uses the [`uv`](https://github.com/astral-sh/uv) package manager for faster and more reliable Python dependency management.

To install `uv`, please refer to their [official installation guide](https://github.com/astral-sh/uv#installation).

After installing `uv`, install all dependencies with:

```bash
uv sync --all-extras --no-install-project

# Activate the virtual env
source .venv/bin/activate
```

This will create a virtual environment, install all required packages, and activate the virtual environment.

This repository uses Weights & Biases (W&B) for logging metrics and visualizations, login to your W&B account:

```bash
wandb login
```

## Dataset 📂

Prepare your Cryo-ET tomograms and corresponding segmentation labels in the following directory structure:

```
data/
  train/
    tomograms/
    labels/
  test/
    tomograms/
```

Update the paths in the config files if needed.

## Training 🏋️‍♀️

To train the base segmentation model on your training dataset:

```bash
python train.py --config configs/train_config.yaml
```

You can adjust hyperparameters and paths in the `configs/train_config.yaml` file.

The trained model checkpoints will be saved in the `checkpoints/` directory.

## Test-time training 📈

Once the base model is trained, you can perform denoising test-time training (TTT) to adapt the model to each new test tomogram:

```bash
python ttt.py --config configs/ttt_config.yaml --checkpoint checkpoints/best_model.pth
```

This will load the pre-trained model and fine-tune it on the test tomogram using self-supervised denoising objectives. The adapted segmentation will be saved to `outputs/`.

You can configure TTT hyperparameters (e.g., adaptation steps, learning rate, denoising loss) in `configs/ttt_config.yaml`.

## Citation 📚

If you use this code for your research, please cite:

```
@article{YourCitation2025,
  title={Denoising Test-Time Training for Membrane Segmentation in Cryo-ET},
  author={Your Name et al.},
  journal={Journal Name},
  year={2025}
}
```

## License ⚖️

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact 📬

For questions or issues, please open an issue or contact [diyor.khayrutdinov@tum.com](mailto:diyor.khayrutdinov@domain.com).
