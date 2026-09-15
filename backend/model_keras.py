try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_keras_densenet(input_shape=(224, 224, 3), num_classes=2):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed in the current Python environment.")

    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(40, activation="relu", name="den")(x)
    x = Dense(20, activation="relu", name="den_1")(x)
    
    if num_classes == 2:
        outputs = Dense(2, activation="softmax", name="den_2")(x)
        loss = "categorical_crossentropy"
    else:
        outputs = Dense(1, activation="sigmoid", name="den_2")(x)
        loss = "binary_crossentropy"

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    
    return model


if __name__ == '__main__':
    if TF_AVAILABLE:
        model = build_keras_densenet()
        model.summary()
    else:
        print("TensorFlow is not installed. Using PyTorch backend in backend/model.py.")
