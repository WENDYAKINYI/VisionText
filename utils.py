# utils.py

import torch
import torch.nn.functional as F
import openai
import numpy as np
from PIL import Image
import os
import requests
import pickle
from io import BytesIO
from models import EncoderCNN, DecoderRNN
from huggingface_hub import hf_hub_download
from yolo_ultra_utils import detect_objects_yolo_ultralytics
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
import requests

# New: Direct model download
def download_model_from_url(url, save_path="best_model.pth"):
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved model to {save_path}")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
    else:
        print(f"Model already exists at {save_path}")
    return save_path

# Updated: Load baseline model from direct URL
def load_baseline_model():
    model_url = "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/best_model.pth"
    checkpoint_path = download_model_from_url(model_url)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab = checkpoint['vocab']
    encoder = EncoderCNN(embed_size=512).to(device)
    decoder = DecoderRNN(embed_size=512, hidden_size=512, vocab_size=len(vocab)).to(device)

    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    decoder.load_state_dict(checkpoint['decoder'], strict=False)

    return encoder, decoder, vocab


def generate_baseline_caption(image_tensor, encoder, decoder, vocab, beam_size=3, max_len=20):
    idx2word = {idx: word for word, idx in vocab.items()}
    word2idx = vocab

    features = encoder(image_tensor)
    encoder_out = features.unsqueeze(1)
    encoder_dim = encoder_out.size(-1)
    num_pixels = encoder_out.size(1)
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)

    seqs = torch.full((beam_size, 1), word2idx['<start>'], dtype=torch.long, device=image_tensor.device)
    top_k_scores = torch.zeros(beam_size, 1, device=image_tensor.device)

    complete_seqs = []
    complete_seqs_scores = []

    h, c = decoder.init_hidden_state(encoder_out.mean(dim=1))

    for step in range(max_len):
        prev_words = seqs[:, -1].unsqueeze(1)
        embeddings = decoder.embed(prev_words)
        context = encoder_out.mean(dim=1)
        lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)
        h, c = decoder.lstm(lstm_input.unsqueeze(1), (h, c))
        scores = F.log_softmax(decoder.linear(h[0]), dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 0:
            top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)

        prev_seq_inds = top_k_words // len(vocab)
        next_word_inds = top_k_words % len(vocab)

        seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, word in enumerate(next_word_inds) if word != word2idx['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())

        if len(incomplete_inds) == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[0][prev_seq_inds[incomplete_inds]].unsqueeze(0), h[1][prev_seq_inds[incomplete_inds]].unsqueeze(0)
        c = c[0][prev_seq_inds[incomplete_inds]].unsqueeze(0), c[1][prev_seq_inds[incomplete_inds]].unsqueeze(0)
        encoder_out = encoder_out[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_seqs_scores = top_k_scores.tolist()

    best_idx = np.argmax(complete_seqs_scores)
    caption_ids = complete_seqs[best_idx]

    caption_words = [idx2word[wid] for wid in caption_ids
                     if idx2word[wid] not in ['<start>', '<end>', '<pad>']]
    confidence = min(float(np.exp(np.max(complete_seqs_scores))), 1.0)

    return {
        "caption": ' '.join(caption_words),
        "confidence": confidence,
        "alphas": []
    }


def enhance_with_openai(caption, max_tokens=100, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Improve this image caption concisely while preserving factual accuracy:"},
                {"role": "user", "content": caption}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return None


def load_image(image_source):
    try:
        if isinstance(image_source, str):
            if image_source.startswith(('http:', 'https:')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                return Image.open(image_source).convert('RGB')
        else:
            return Image.open(image_source).convert('RGB')
    except Exception as e:
        print(f"Image load failed: {str(e)}")
        return None


def preprocess_image(image, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def get_detected_objects(image_tensor, conf_thresh=0.5):
    try:
        return detect_objects_yolo_ultralytics(image_tensor, conf_thresh=conf_thresh)
    except Exception as e:
        print(f"YOLO detection failed: {str(e)}")
        return []
