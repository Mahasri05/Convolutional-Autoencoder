# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Image denoising is the process of removing noise from images while preserving important features. Noise may occur due to low light, sensor issues, or transmission errors.

In this experiment, a Convolutional Autoencoder (CAE) is used to learn how to reconstruct clean images from noisy images.

The MNIST dataset is used for training and testing. MNIST consists of 70,000 grayscale images of handwritten digits (0–9) with a size of 28 × 28 pixels.

Noise is artificially added to the images, and the autoencoder learns to recover the original image.
## DESIGN STEPS

### STEP 1:

Import the required libraries such as TensorFlow, Keras, NumPy, and Matplotlib.

### STEP 2:

Load the MNIST dataset and normalize the image pixel values between 0 and 1.

### STEP 3:

Add random noise to the dataset to create noisy images.

### STEP 4:

Build the Convolutional Autoencoder architecture consisting of:

Encoder (Convolution + MaxPooling layers)

Decoder (Convolution + UpSampling layers)

### STEP 5:

Compile the model using Adam optimizer and binary cross-entropy loss function.

### STEP 6:

Train the model using the noisy images as input and original images as output.

### STEP 7:

Test the model using noisy test images and generate reconstructed images.

### STEP 8:

Display Original Image, Noisy Image, and Reconstructed Image for comparison.


## PROGRAM
### Name:Mahasri D
### Register Number:212224220058
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in loader:

            images = images.to(device)

            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
def visualize_denoising(model, loader, num_images=5):

    model.eval()

    with torch.no_grad():
        for images, _ in loader:

            images = images.to(device)
            noisy_images = add_noise(images)

            outputs = model(noisy_images)
            

            images = images.cpu()
            noisy_images = noisy_images.cpu()
            outputs = outputs.cpu()
            print("Name:Mahasri D")
            print("Register Number:212224220058")
            plt.figure(figsize=(10,4))

            for i in range(num_images):

                # Original
                plt.subplot(3, num_images, i+1)
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title("Original")
                plt.axis('off')

                # Noisy
                plt.subplot(3, num_images, i+1+num_images)
                plt.imshow(noisy_images[i].squeeze(), cmap='gray')
                plt.title("Noisy")
                plt.axis('off')

                # Reconstructed
                plt.subplot(3, num_images, i+1+2*num_images)
                plt.imshow(outputs[i].squeeze(), cmap='gray')
                plt.title("Denoised")
                plt.axis('off')
                
            plt.show()
            break
train(model, train_loader, criterion, optimizer, epochs=5)

visualize_denoising(model, test_loader)
```


## OUTPUT
<img width="585" height="355" alt="image" src="https://github.com/user-attachments/assets/1777a0ba-5a93-4726-9b67-ee7cfcb48eff" />


### Model Summary

<img width="686" height="508" alt="image" src="https://github.com/user-attachments/assets/ca14382b-ade7-4a60-865c-249c240d5c2a" />


### Original vs Noisy Vs Reconstructed Image
<img width="720" height="489" alt="image" src="https://github.com/user-attachments/assets/d8a3f3bf-59e1-4657-910a-aeebd84c0dd2" />




## RESULT
Thus, the Convolutional Autoencoder model was successfully developed and trained for image denoising, and the reconstructed images closely resemble the original images by removing noise.
