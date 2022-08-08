import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import data_parse
from data_parse import parse_dataset

os.environ["CUDA_VISIBLE_DEVICES"]="1"

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

print("FINISHED AUGMENT")

def conv_layer(x, filters):
    x = layers.Conv1D(filters, kernel_size = 1, padding = "valid")(x)
    x = layers.BatchNormalization(momentum = 0.0)(x)
    return layers.Activation("relu")(x)
    
def dense_layer(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum = 0.0)(x)
    return layers.Activation("relu")(x)
    
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_layer(inputs, 32)
    x = conv_layer(x, 64)
    x = conv_layer(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_layer(x, 256)
    x = dense_layer(x, 128)
    x = layers.Dense( num_features * num_features, kernel_initializer = "zeros", bias_initializer = bias, activity_regularizer=reg)(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    print("TNET HAS BEEN CREATED")
    
    #Apply Affine Transform to input features
    return layers.Dot(axes = (2,1))([inputs, feat_T])

print("Initiated training")
inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_layer(x, 32)
x = conv_layer(x, 32)
print("layer1 done")
x = tnet(x, 32)
x = conv_layer(x, 32)
x = conv_layer(x, 64)
x = conv_layer(x, 512)
x = layers.GlobalMaxPooling1D()(x)
print("layer2 done")
x = dense_layer(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_layer(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

#TRAIN MODEL
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:15, ...]
labels = labels[:15, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], c ='g', s=2)
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("plot:" + str(i) +".png", format = "PNG" )
plt.show()