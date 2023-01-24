import pathlib
import matplotlib.pyplot as plt
import utils
from trainer import Trainer

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(
        trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(
        trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    
def create_comparison_plots(trainer_1: Trainer, trainer_2: Trainer, name:str):
    """
    Function for generating comparison plots with different trained models
    """
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer_1.train_history["loss"], label="Train loss before", npoints_to_average=10)
    utils.plot_loss(trainer_2.train_history["loss"], label="Train loss after", npoints_to_average=10)
    utils.plot_loss(trainer_1.validation_history["loss"], label="Val loss before")
    utils.plot_loss(trainer_2.validation_history["loss"], label="Val loss after")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer_1.validation_history["accuracy"], label="Val Accuracy before")
    utils.plot_loss(trainer_2.validation_history["accuracy"], label="Val Accuracy after")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
