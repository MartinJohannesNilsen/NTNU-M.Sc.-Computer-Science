import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    ############### TASK RUNNER ###############
    # 1 - Shuffled vs unshuffled              #
    # 2 - Tricks of trade                     #
    # 3 - Too small/large hidden units        #
    # 4 - Network with 2 hidden layers        #
    # 5 - Network with 10 hidden layers       #
    ###########################################
    # Comment: As data shuffling is generally good, I will keep it in the other tasks as well

    SELECTED_TASK = 4
    SHOW_PLOT = True
    SAVE_PLOT = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    if SELECTED_TASK == 1:
        shuffle_data = True
        use_improved_sigmoid = False
        use_improved_weight_init = False
        use_momentum = False

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_shuffled, val_history_shuffled = trainer.train(num_epochs)

        shuffle_data = False
        model_no_shuffle = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_no_shuffle, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_unshuffled, val_history_unshuffled = trainer_shuffle.train(
            num_epochs)

        plt.figure(figsize=(20, 12))
        plt.subplot(1, 2, 1)
        utils.plot_loss(train_history_shuffled["loss"],
                        "Shuffled", npoints_to_average=10)
        utils.plot_loss(
            train_history_unshuffled["loss"], "Unshuffled", npoints_to_average=10)
        utils.plot_loss(val_history_shuffled["loss"], "Validation Loss Shuffled")
        utils.plot_loss(val_history_unshuffled["loss"], "Validation Loss Unshuffled")
        plt.ylabel("Loss")
        plt.ylim([0, .4])
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.ylim([0.89, .95])
        utils.plot_loss(
            val_history_shuffled["accuracy"], "Shuffled")
        utils.plot_loss(
            val_history_unshuffled["accuracy"], "Unshuffled")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        if SAVE_PLOT:
            plt.savefig("../report/Images/task3_shuffled.png")
        if SHOW_PLOT:
            plt.show()

    if SELECTED_TASK == 2:
        shuffle_data = True
        use_improved_weight_init = False
        use_improved_sigmoid = False
        use_momentum = False

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_shuffled, val_history_shuffled = trainer.train(num_epochs)

        use_improved_weight_init = True
        model_no_shuffle = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_no_shuffle, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_weights, val_history_weights = trainer_shuffle.train(
            num_epochs)

        use_improved_sigmoid = True
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        use_momentum = True
        learning_rate_momentum = 0.02
        model_no_shuffle = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_no_shuffle, learning_rate_momentum, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_final, val_history_final = trainer_shuffle.train(
            num_epochs)

        plt.figure(figsize=(20, 12))
        plt.subplot(1, 2, 1)
        utils.plot_loss(train_history["loss"], "Training Loss Using Improved Sigmoid", npoints_to_average=10)
        utils.plot_loss(train_history_weights["loss"], "Training Loss Using Improved Weights", npoints_to_average=10)
        utils.plot_loss(train_history_final["loss"], "Training Loss Using Momentum", npoints_to_average=10)
        utils.plot_loss(val_history["loss"], "Validation Loss Using Improved Sigmoid")
        utils.plot_loss(val_history_weights["loss"], "Validation Loss Using Improved Weights")
        utils.plot_loss(val_history_final["loss"], "Validation Loss Using Momentum")
        plt.ylabel("Cross entropy Loss")
        plt.ylim([0, .4])
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.ylim([0.89, .99])
        utils.plot_loss(val_history_shuffled["accuracy"], "Validation Loss Using Only Shuffling")
        utils.plot_loss(val_history["accuracy"], "Validation Loss Using Improved Sigmoid")
        utils.plot_loss(val_history_weights["accuracy"], "Validation Loss Using Improved Weights")
        utils.plot_loss(val_history_final["accuracy"], "Validation Loss Using Using Momentum")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        if SAVE_PLOT:
            plt.savefig("../report/Images/task3_tricks.png")
        if SHOW_PLOT:
            plt.show()

    if SELECTED_TASK in [3, 4, 5]:  # 4a/b but also used in 4d and 4e, therefore [3,4,5]
        if use_momentum:
            learning_rate = 0.02  # momentum requires different learning rate

        # 32 hidden units
        neurons_per_layer = [32, 10]
        model_32 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_32, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_32, val_history_32 = trainer_shuffle.train(num_epochs)

        # 64 hidden units
        neurons_per_layer = [64, 10]
        model_64 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_64, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_64, val_history_64 = trainer_shuffle.train(num_epochs)

        # 128 hidden units
        neurons_per_layer = [128, 10]
        model_128 = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_128, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_128, val_history_128 = trainer_shuffle.train(num_epochs)

        if SELECTED_TASK == 3:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0.00, 0.4])
            utils.plot_loss(train_history_32["loss"], "32 hidden units", npoints_to_average=10)
            utils.plot_loss(train_history_64["loss"], "64 hidden units", npoints_to_average=10)
            utils.plot_loss(train_history_128["loss"], "128 hidden units", npoints_to_average=10)
            utils.plot_loss(val_history_32["loss"], "32 hidden units, validation loss")
            utils.plot_loss(val_history_64["loss"], "64 hidden units, validation loss")
            utils.plot_loss(val_history_128["loss"], "128 hidden units, validation loss")
            plt.ylabel("Training Loss")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.ylim([0.93, .98])
            utils.plot_loss(val_history_32["accuracy"], "32 hidden units")
            utils.plot_loss(val_history_64["accuracy"], "64 hidden units")
            utils.plot_loss(val_history_128["accuracy"], "128 hidden units")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            if SAVE_PLOT:
                plt.savefig("../report/Images/task4ab.png")
            if SHOW_PLOT:
                plt.show()

    if SELECTED_TASK in [4, 5]:  # 4d, but using this in 4e too, so 4 and 5
        # Tricks of trade from 3
        neurons_per_layer = [60, 60, 10]
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        if SELECTED_TASK == 4:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            utils.plot_loss(train_history_64["loss"], "Training Loss [64, 10]", npoints_to_average=10)
            utils.plot_loss(val_history_64["loss"], "Validation Loss [64, 10]")
            utils.plot_loss(train_history["loss"], f"Training Loss {neurons_per_layer}", npoints_to_average=10)
            utils.plot_loss(val_history["loss"], f"Validation Loss {neurons_per_layer}")
            plt.ylabel("Loss")
            plt.ylim([0, .4])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.ylim([0.92, .98])
            utils.plot_loss(val_history_64["accuracy"], f"Validation Accuracy [64, 10]")
            utils.plot_loss(val_history["accuracy"], f"Validation Accuracy {neurons_per_layer}")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            if SAVE_PLOT:
                plt.savefig("../report/Images/task4d_2hidden.png")
            if SHOW_PLOT:
                plt.show()

    if SELECTED_TASK == 5:  # 4e
        # Tricks of trade from 3

        neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
        model_10_layers = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_shuffle = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_10_layers, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_10_layers, val_history_10_layers = trainer_shuffle.train(num_epochs)

        plt.figure(figsize=(20, 12))
        plt.subplot(1, 2, 1)
        plt.ylim([0.00, 1.00])
        utils.plot_loss(train_history_10_layers["loss"], "Training Loss - 10 hidden layers", npoints_to_average=10)
        plt.ylabel("Training Loss")
        utils.plot_loss(val_history_10_layers["loss"], "Validation Loss - 10 hidden layers")
        plt.ylabel("Validation Loss")
        utils.plot_loss(train_history["loss"], "Training Loss - 2 hidden layers", npoints_to_average=10)
        utils.plot_loss(val_history["loss"], "Validation Loss - 2 hidden layers")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.ylim([0.86, .97])
        utils.plot_loss(val_history_10_layers["accuracy"], "Validation Accuracy - 10 hidden layers")
        utils.plot_loss(val_history["accuracy"], "Validation Accuracy - 2 hidden layers")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        if SAVE_PLOT:
            plt.savefig("../report/Images/task4e_10hidden.png")
        if SHOW_PLOT:
            plt.show()
