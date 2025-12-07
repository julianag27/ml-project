from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd

def nn_run_models(X_train, Y_train, X_test, Y_test):
    # Initializing final stats table
    stats = pd.DataFrame({
        "label": [],
        "desc": [],
        "model_type": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": [],
        "ROC/AUC": []
    })

    models = nn_get_models(X_train.shape)
    for label, desc, model, callbacks in models:
        eval = None
        if 'softmax' in label:
            model.fit(x=X_train, y=to_categorical(Y_train), batch_size=128, epochs=10, validation_split=0.2, callbacks=callbacks, verbose=0)
            eval = model.evaluate(X_test, to_categorical(Y_test), verbose=0)
            tn = fp = fn = tp = 0
            pred = model.predict(X_test, verbose=0)
            for true, (_, one) in zip(Y_test, pred):
                if true == 1 and one >= 0.5:
                    tp += 1
                if true == 1 and one < 0.5:
                    fn += 1
                if true == 0 and one >= 0.5:
                    fp += 1
                if true == 0 and one < 0.5:
                    tn += 1
            eval[5] = tn
            eval[6] = fp
            eval[7] = fn
            eval[8] = tp
        else:
            model.fit(x=X_train, y=Y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=callbacks, verbose=0)
            eval = model.evaluate(X_test, Y_test, verbose=0)
            train_eval = model.evaluate(X_train, Y_train, verbose=0)

        # Generate statistics
        tn = int(eval[5])
        fp = int(eval[6])
        fn = int(eval[7])
        tp = int(eval[8])
        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["Neural Net"],
            "train_accuracy": [train_eval[1]],
            "test_accuracy": [(tn+tp)/(tn+fp+fn+tp)],
            "precision": [tp/(tp+fp)],
            "recall": [tp/(tp+fn)],
            "f1_score": [tp/(tp + ((fp+fn)/2))],
            "confusion_matrix": [[[tn, fp], [fn, tp]]],
            "ROC/AUC": [eval[9]]
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
        "true_positives",
        "auc"
    ]
    # add models as functions and add the function to the list below
    return [
        leaky_adam(metrics, X_shape),
        leaky_adam_x5(metrics, X_shape),
        leaky_adam_x5_no_drop(metrics, X_shape),
        leaky_adam_1024(metrics, X_shape),
        leaky_adam_bin_crossent(metrics, X_shape),
        elu_adam_bin_crossent(metrics, X_shape),
        sigmoid_adam_bin_crossent(metrics, X_shape),
        lin_adam_bin_crossent(metrics, X_shape),
        tanh_adam_bin_crossent(metrics, X_shape),
        leaky_adam_cat_cross(metrics, X_shape),
        leaky_adam_softmax(metrics, X_shape),
        tanh_adam_1024_noearly(metrics, X_shape),
        tanh_adam_1024_noearly_softmax(metrics, X_shape),
        tanh_adam_1024_2_noearly_softmax(metrics, X_shape)
    ] 

def leaky_adam(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
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

def leaky_adam_x5(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, binary_crossentropy, 5 layers"
    desc ="\tActivation function(s): leaky_relu, sigmoid\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def leaky_adam_x5_no_drop(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, binary_crossentropy, 5 layers, no dropout"
    desc ="\tActivation function(s): leaky_relu, sigmoid\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\t64, leaky_relu\n\t\t64, leaky_relu\n\t\t64, leaky_relu\n\t\t64, leaky_relu\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def leaky_adam_1024(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(1024, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, 1024y"
    desc ="\tActivation function(s): leaky_relu, sigmoid\n" \
        "\tLayers:\n\t\t1024, leaky_relu\n\t\tDropout: 0.2\n\t\t256, leaky_relu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 1024\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def leaky_adam_bin_crossent(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, binary_crossentropy, no_early_stop"
    desc ="\tActivation function(s): leaky_relu, sigmoid\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss)"
    callbacks = [reduce_lr]
    return (label, desc, model, callbacks)

def elu_adam_bin_crossent(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "elu, adam, binary_crossentropy"
    desc ="\tActivation function(s): elu, sigmoid\n" \
        "\tLayers:\n\t\t128, elu\n\t\tDropout: 0.2\n\t\t64, elu\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss)"
    callbacks = [reduce_lr]
    return (label, desc, model, callbacks)

def sigmoid_adam_bin_crossent(metrics, X_shape):

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "sigmoid, adam, binary_crossentropy"
    desc ="\tActivation function(s): sigmoid\n" \
        "\tLayers:\n\t\t128, sigmoid\n\t\tDropout: 0.2\n\t\t64, sigmoid\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def lin_adam_bin_crossent(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "linaer, adam, binary_crossentropy"
    desc ="\tActivation function(s): linear, sigmoid\n" \
        "\tLayers:\n\t\t128, linear\n\t\tDropout: 0.2\n\t\t64, linear\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def tanh_adam_bin_crossent(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "tanh, adam, binary_crossentropy"
    desc ="\tActivation function(s): tanh, sigmoid\n" \
        "\tLayers:\n\t\t128, tanh\n\t\tDropout: 0.2\n\t\t64, tanh\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def leaky_adam_cat_cross(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
    
    label = "leaky_relu, adam, categorical_crossentropy, softmax"
    desc ="\tActivation function(s): leaky_relu, softmax\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t2, softmax\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: categorical_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def leaky_adam_softmax(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "leaky_relu, softmax, adam, binary_crossentropy"
    desc ="\tActivation function(s): leaky_relu, softmax\n" \
        "\tLayers:\n\t\t128, leaky_relu\n\t\tDropout: 0.2\n\t\t64, leaky_relu\n\t\tDropout: 0.2\n\t\t2, softmax\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 128\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr, early_stopping]
    return (label, desc, model, callbacks)

def tanh_adam_1024_noearly(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "tahn, adam, 1024, noearly"
    desc ="\tActivation function(s): tanh, sigmoid\n" \
        "\tLayers:\n\t\t1024, tanh\n\t\tDropout: 0.2\n\t\t256, tanh\n\t\tDropout: 0.2\n\t\t1, sigmoid\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 1024\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr]
    return (label, desc, model, callbacks)

def tanh_adam_1024_noearly_softmax(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "tahn, adam, 1024, noearly, softmax"
    desc ="\tActivation function(s): tanh, softmax\n" \
        "\tLayers:\n\t\t1024, tanh\n\t\tDropout: 0.2\n\t\t256, tanh\n\t\tDropout: 0.2\n\t\t2, softmax\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 1024\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr]
    return (label, desc, model, callbacks)

def tanh_adam_1024_2_noearly_softmax(metrics, X_shape):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    model = Sequential()

    # Add layers
    model.add(Input(shape=(X_shape[1],)))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    label = "tahn, adam, 1024x2, noearly, softmax"
    desc ="\tActivation function(s): tanh, softmax\n" \
        "\tLayers:\n\t\t1024, tanh\n\t\tDropout: 0.2\n\t\t1024, tanh\n\t\tDropout: 0.2\n\t\t256, tanh\n\t\tDropout: 0.2\n\t\t2, softmax\n" \
        "\tOptimizer: adam\n" \
        "\tLoss: binary_crossentropy\n" \
        "\tEpochs: 10\n" \
        "\tBatch Size: 1024\n" \
        "\tCallbacks: ReduceLR (val_loss), EarlyStopping (val_loss)"
    callbacks = [reduce_lr]
    return (label, desc, model, callbacks)
