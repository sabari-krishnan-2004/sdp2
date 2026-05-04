import tensorflow as tf
from config import NUM_CLASSES

# 1. Load your central dataset
print("Loading central dataset for pre-training...")
central_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/central_data", # Make sure you have images here!
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
central_ds = central_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. Build the Fresh MobileNet Model
print("Building new MobileNet architecture...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 3. Pre-train the model
print("Starting Centralized Pre-training...")
model.fit(central_ds, epochs=25) # Train for 5-10 epochs to get a good baseline

# 4. Save the "Warm" Global Model
model.save("pretrained_global_model.h5")
print("✅ Success! 'pretrained_global_model.h5' is ready for Federated Learning.")