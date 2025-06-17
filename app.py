# === Imports ===
import streamlit as st                     # For creating the web UI
import os                                  # For interacting with the file system
import torch                               # PyTorch: deep learning framework
import clip                                # OpenAI's CLIP model for vision-text embeddings
from PIL import Image                      # For image loading/processing
import numpy as np                         # Numerical operations
from sklearn.metrics.pairwise import cosine_similarity  # To compare embeddings
import cv2                                 # For image resizing and basic manipulation
import base64                              # For encoding image to allow download
from io import BytesIO                     # For creating image buffer to serve as downloadable content

# === App Config ===
st.set_page_config(page_title="Who Do You Look Like?", layout="centered")
st.title("Who Do You Look Like?")
st.caption("Upload a selfie, and AI finds your top 3 celeb twins!")

# === Load CLIP model (ViT-B/32) === (324 MB)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === Load all celebrity images and pre-compute their embeddings ===
@st.cache_data  
# Cache to avoid recomputation every time
def load_celeb_embeddings(folder):
    names, images, embeddings = [], [], []
    for file in os.listdir(folder):
        if file.endswith((".jpg", ".png")):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("RGB")  
            # Open image
            names.append(os.path.splitext(file)[0].replace("_", " "))  
            # Store celeb name
            images.append(img)
            # Encode celeb image into embedding
            with torch.no_grad():
                emb = model.encode_image(preprocess(img).unsqueeze(0).to(device))
            embeddings.append(emb.cpu().numpy())
    return names, images, np.vstack(embeddings)  
# Return all as numpy arrays

# === Load celeb data ===
names, celeb_imgs, celeb_embeds = load_celeb_embeddings("celeb_images")

# === Generate a horizontal collage and return a downloadable link ===
def generate_collage_and_link(selfie_img, celeb_img, celeb_name):
    # Resize both to same dimensions
    selfie = cv2.resize(np.array(selfie_img), (300, 300))
    celeb = cv2.resize(np.array(celeb_img), (300, 300))
    collage = np.hstack((selfie, celeb))  
    # Stack side-by-side

    # Convert to PIL to enable download
    collage_pil = Image.fromarray(collage)
    buf = BytesIO()
    collage_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode()
    
    # Download link
    href = f'<a href="data:file/png;base64,{b64}" download="your_lookalike.png">📥 Download Result Image</a>'
    return collage_pil, href

# === File uploader widget ===
uploaded = st.file_uploader("Upload your selfie", type=["jpg", "png"])

# === Main logic: if selfie uploaded ===
if uploaded:
    # Load and display selfie
    selfie = Image.open(uploaded).convert("RGB")

    # Encode selfie to embedding
    with torch.no_grad():
        self_emb = model.encode_image(preprocess(selfie).unsqueeze(0).to(device)).cpu().numpy()

    # Compare to all celeb embeddings using cosine similarity
    sims = cosine_similarity(self_emb, celeb_embeds)[0]
    top_indices = np.argsort(sims)[::-1][:3]  
    # Top 3 matches

    # === Show top 3 matches ===
    st.markdown("### Top 3 Matches")
    cols = st.columns(3)
    for i, idx in enumerate(top_indices):
        name = names[idx]
        score = sims[idx] * 100  
        # Convert to percentage
        celeb_img = celeb_imgs[idx]
        with cols[i]:
            st.image(celeb_img, caption=f"{name}\n{score:.2f}%", width=200)

    # === Pick best match ===
    best_idx = top_indices[0]
    best_name = names[best_idx]
    best_img = celeb_imgs[best_idx]
    similarity_score = sims[best_idx] * 100

    st.markdown("---")
    st.markdown(f"### ✅ Best Match: **{best_name}**")
    st.write(f"**Confidence Score:** {similarity_score:.2f}%")

    # === Explanation (currently static) ===
    st.info(
        f"You and **{best_name}** both share similar facial structure, expression, and lighting style."
    )

    # === Show side-by-side collage and download button ===
    st.markdown("### 📸 Side-by-Side Comparison")
    collage, download_link = generate_collage_and_link(selfie, best_img, best_name)
    st.image(collage, caption="Left: You | Right: Match", width=600)
    st.markdown(download_link, unsafe_allow_html=True)
