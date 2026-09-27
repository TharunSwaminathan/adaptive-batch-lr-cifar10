import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    EPOCHS,
    INITIAL_BATCH_SIZE,
    INITIAL_LEARNING_RATE,
    MOMENTUM,
    OPTIMIZER,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    print_config,
    set_seed,
)

from data.cifar10 import (
    CIFAR10DataModule,
)

from models.custom_cnn import (
    CustomCNN,
)

from training.trainer import (
    Trainer,
)


def main():

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    set_seed(
        SEED
    )


    # -----------------------------------------------------
    # Show experiment configuration
    # -----------------------------------------------------

    print_config()


    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    data = CIFAR10DataModule()


    train_loader = (
        data.get_train_loader(
            INITIAL_BATCH_SIZE
        )
    )


    val_loader = (
        data.get_val_loader(
            batch_size=256
        )
    )


    # IMPORTANT:
    #
    # We intentionally do NOT create/use the test loader
    # during this pilot.
    #
    # The test set stays untouched until the experiment
    # configuration has been finalized.


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = CustomCNN().to(
        DEVICE
    )


    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )


    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    if OPTIMIZER.lower() == "sgd":

        optimizer = optim.SGD(
            model.parameters(),

            lr=INITIAL_LEARNING_RATE,

            momentum=MOMENTUM,

            weight_decay=WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{OPTIMIZER}"
        )


    # -----------------------------------------------------
    # Run name
    # -----------------------------------------------------

    run_name = (
        f"pilot_fixedBatch"
        f"{INITIAL_BATCH_SIZE}"
        f"_fixedLR"
        f"{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )


    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(

        model=model,

        criterion=criterion,

        optimizer=optimizer,

        device=DEVICE,

        results_dir=RESULTS_DIR,

        checkpoint_dir=CHECKPOINT_DIR,

        run_name=run_name,
    )


    # -----------------------------------------------------
    # Fixed baseline pilot
    # -----------------------------------------------------

    trainer.fit(

        train_loader=train_loader,

        val_loader=val_loader,

        epochs=EPOCHS,

        batch_size=INITIAL_BATCH_SIZE,
    )


if __name__ == "__main__":

    main()