import streamlit as st
import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.optim as optim
from utils import image_loader, im_convert, gram_matrix, get_features
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cpu")


st.title("🎨 Neural Style Transfer in Your Browser")
st.write("Upload a **content** and **style** image, then click Run!")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Save uploaded files temporarily
    content_path = "images/content.jpg"
    style_path = "images/style.jpg"
    os.makedirs("images", exist_ok=True)
    with open(content_path, "wb") as f:
        f.write(content_file.read())
    with open(style_path, "wb") as f:
        f.write(style_file.read())

    # Load images
    content = image_loader(content_path, device)
    style = image_loader(style_path, device, shape=content.shape[-2:])

    # Load model
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    # Extract features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target image
    target = content.clone().requires_grad_(True).to(device)

    # Style weights
    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.1
    }
    content_weight = 1e4
    style_weight = 1e2

if st.button("Run Style Transfer"):
        optimizer = optim.Adam([target], lr=0.003)
        epochs = 300
        progress = st.progress(0)
        for i in range(epochs):
            target_features = get_features(target, vgg)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

            style_loss = 0
            for layer in style_weights:
                target_gram = gram_matrix(target_features[layer])
                style_gram = style_grams[layer]
                layer_loss = torch.mean((target_gram - style_gram)**2)
                style_loss += style_weights[layer] * layer_loss / (target_features[layer].shape[1]**2)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                progress.progress(i / epochs)

        progress.progress(1.0)  # complete progress bar

        final_img = im_convert(target)
        os.makedirs("output", exist_ok=True)
        output_path = "output/stylized_output.jpg"
        plt.imsave(output_path, final_img)

        st.image(final_img, caption="Styled Output", use_column_width=True)
        st.write("Running style transfer... Please wait.")




