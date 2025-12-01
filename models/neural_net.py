from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd


def nn_run_models(X_train, Y_train, X_test, Y_test):
    # Initializing final stats table
    stats = pd.DataFrame({
        "label": [],
        "desc": [],
        "model_type": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": []
    })

    models = nn_get_models(X_train.shape)
    for label, desc, model, callbacks in models:
        history = model.fit(x=X_train, y=Y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=callbacks, verbose=0)

        # Generate statistics
        eval = model.evaluate(X_test, Y_test, verbose=0)
        print(eval)
        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["Neural Net"],
            "accuracy": [eval[1]],
            "precision": [eval[2]],
            "recall": [eval[3]],
            "f1_score": [eval[4]],
            "confusion_matrix": [[[[eval[5], eval[6]]], [eval[7], eval[8]]]]
        })], ignore_index=True)
    return stats

def nn_get_models(X_shape):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives"
    ]
    # add models as functions and add the function to the list below
    return [
        leaky_adam_bin_crossent(metrics, X_shape)
    ] 

def leaky_adam_bin_crossent(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, binary_crossentropy"
    desc ="\tActivation function(s): leaky_relu, sigmoid\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)