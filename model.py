import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import safetensors


print(tf.config.list_physical_devices('GPU'))

# Define paths to your datasets
routing_path = Path("C:" + os.sep + "Users" + os.sep + "suran" + os.sep + "Desktop" + os.sep + "School" + os.sep +
                "1_UNIVERSITY" + os.sep + "BENNETT" + os.sep + "6thSem" + os.sep + "IVP" + os.sep + "SmartPlanAI")
color_dir = Path(str(routing_path) + os.sep + "Data" + os.sep + "Database" + os.sep + "colors")
walls_dir = Path(str(routing_path) + os.sep + "Data" + os.sep +"Database" + os.sep + "walls")
save_dir = Path(str(routing_path) + os.sep + "Model")
pretrained_model_path = Path(str(save_dir) + os.sep + "instruct-pix2pix-00-22000.safetensors")

# Define image transformations
transform = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)


def create_dataset(color_dir, walls_dir, transform):
    color_paths = tf.io.gfile.glob(str(color_dir / "*.png"))
    walls_paths = tf.io.gfile.glob(str(walls_dir / "*.png"))
    color_images = tf.map_fn(lambda x: tf.io.decode_png(tf.io.read_file(x), channels=3), color_paths, dtype=tf.float32)
    walls_images = tf.map_fn(lambda x: tf.io.decode_png(tf.io.read_file(x), channels=3), walls_paths, dtype=tf.float32)
    color_images = transform(color_images)
    walls_images = transform(walls_images)
    dataset = tf.data.Dataset.zip((color_images, walls_images))
    return dataset


# Define the Generator network with U-Net architecture
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Freeze the pretrained model weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Define U-Net encoder layers
        self.down1 = layers.Conv2D(3, 64, kernel_size=4, stride=2, padding="same")
        self.down2 = layers.Conv2D(64, 128, kernel_size=4, stride=2, padding="same")
        self.down3 = layers.Conv2D(128, 256, kernel_size=4, stride=2, padding="same")
        self.down4 = layers.Conv2D(256, 512, kernel_size=4, stride=2, padding="same")
        self.down5 = layers.Conv2D(512, 512, kernel_size=4, stride=2, padding="same")
        # Define U-Net decoder layers
        self.up1 = layers.Conv2DTranspose(512, 512, kernel_size=4, stride=2, padding="same")
        self.up2 = layers.Conv2DTranspose(1024, 256, kernel_size=4, stride=2, padding="same")
        self.up3 = layers.Conv2DTranspose(512, 128, kernel_size=4, stride=2, padding="same")
        self.up4 = layers.Conv2DTranspose(256, 64, kernel_size=4, stride=2, padding="same")
        self.final_layer = layers.Conv2DTranspose(128, 1, kernel_size=4, stride=2, padding="same")

    def forward(self, inputs):
        # Pass features through U-Net encoder
        d1 = self.down1(inputs)
        d2 = self.down2(layers.LeakyReLU(0.2)(d1))
        d3 = self.down3(layers.LeakyReLU(0.2)(d2))
        d4 = self.down4(layers.LeakyReLU(0.2)(d3))
        d5 = self.down5(layers.LeakyReLU(0.2)(d4))

        # Pass through U-Net decoder
        u1 = self.up1(layers.LeakyReLU(0.2)(d5))
        u2 = self.up2(layers.LeakyReLU(0.2)(layers.Concatenate()([u1, d4])))
        u3 = self.up3(layers.LeakyReLU(0.2)(layers.Concatenate()([u2, d3])))
        u4 = self.up4(layers.LeakyReLU(0.2)(layers.Concatenate()([u3, d2])))
        return self.final_layer(layers.LeakyReLU(0.2)(layers.Concatenate()([u4, d1])))


# Define the Discriminator network with PatchGAN architecture
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0, 0.02)
        self.model = keras.Sequential(
            [
                layers.Conv2D(4, 64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(0.2),
                layers.Conv2D(64, 128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Conv2D(128, 256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Conv2D(256, 512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer, use_bias=False),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Conv2D(512, 1, kernel_size=4, strides=1, padding="same", kernel_initializer=initializer)
            ]
        )

    def forward(self, inputs, conditioning):
        # Concatenate image and condition
        x = layers.Concatenate()([inputs, conditioning])
        return self.model(x)


# Define loss functions and optimizers
bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l1_loss_fn = tf.keras.losses.MeanAbsoluteError()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.999)


#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

# Training loop
  # Convert to TensorFlow graph for efficiency
def train_step(colors, walls, generator, discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake walls
        fake_walls = generator(colors, training=True)

        # Discriminator outputs
        disc_real_output = discriminator([walls, colors], training=True)
        disc_fake_output = discriminator([fake_walls, colors], training=True)

        # Generator loss
        gen_loss = bce_loss_fn(tf.ones_like(disc_fake_output), disc_fake_output) + \
                   100 * l1_loss_fn(walls, fake_walls)

        # Discriminator loss
        real_loss = bce_loss_fn(tf.ones_like(disc_real_output), disc_real_output)
        fake_loss = bce_loss_fn(tf.zeros_like(disc_fake_output), disc_fake_output)
        disc_loss = (real_loss + fake_loss) / 2

    # Calculate gradients and apply updates
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(num_epochs, dataset, batch_size, generator, discriminator):
    dataloader = dataset.batch(batch_size)
    for epoch in range(num_epochs):
        for batch, (colors, walls) in enumerate(dataloader):
            gen_loss, disc_loss = train_step(colors, walls)
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {batch}/{len(dataloader)}] [D loss: {disc_loss.numpy()}] [G loss: {gen_loss.numpy()}]")
            # Save generated images (consider using TensorFlow's image saving utilities)
            batches_done = epoch * len(dataloader) + batch
            if batches_done % 500 == 0:
                generator.save_weights(os.path.join(save_dir, f"generator_{batches_done}.h5"))
                discriminator.save_weights(os.path.join(save_dir, f"discriminator_{batches_done}.h5"))


physical_devices = tf.config.experimental.list_physical_devices('GPU')

# Set the maximum amount of VRAM to be used by the GPU
tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])


# Train the model for a specified number of epochs
with tf.device('/GPU:0'):
    loaded_model = safetensors.torch.load_file(str(pretrained_model_path))
    generator = Generator()
    discriminator = Discriminator()
    num_epochs = 10
    dataset = create_dataset(color_dir, walls_dir, transform=transform)
    batch_size = 1
    train(num_epochs, dataset, batch_size, generator, discriminator)

# Save the trained model
torch.save_file(os.path.join(save_dir, "post_trained_model.h5"))