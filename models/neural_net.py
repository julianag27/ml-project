from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def nn_run_models(X_train, Y_train, X_test, Y_test):
    # Initializing final stats dict
    stats = dict()
    stats["label"] = []
    stats["desc"] = []
    stats["accuracy"] = []
    stats["precision"] = []
    stats["recall"] = []
    stats["f1_score"] = []
    stats["confusion_matrix"] = []

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives"
    ])
    history = model.fit(x=X_train, y=Y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[reduce_lr, early_stopping])

    # Generate statistics
    eval = model.evaluate(X_test, Y_test)
    stats["label"].append("leaky_relu, adam, binary_crossentropy")
    stats["desc"].append(
        "\tActivation function(s): leaky_relu, sigmoid\n\t"
        "Layers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n\t"
        "Optimizer: adam\n\t"
        "Loss: binary_crossentropy\n\t"
        "Epochs: 10\n\t"
        "Batch Size: 128\n\t"
    )
    stats["accuracy"].append(eval[0])
    stats["precision"].append(eval[1])
    stats["recall"].append(eval[2])
    stats["f1_score"].append(eval[3])
    stats["confusion_matrix"].append([[eval[4], eval[5]], [eval[6], eval[7]]])
    return stats