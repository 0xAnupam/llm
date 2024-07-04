

# !pip install torch transformers diffusers  accelerate

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import accelerate

# Loading the tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder = text_encoder.to(device)

# Loading the stable diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 )

pipeline.to(device)



pip install matplotlib



!pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import numpy as np







import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import time

def image_difference(img1, img2):
    """Calculate the difference between two images using PSNR and SSIM"""
    img1_np = np.array(img1).astype(np.float32) / 255.0  # Normalize img1 to [0, 1]
    img2_np = np.array(img2).astype(np.float32) / 255.0  # Normalize img2 to [0, 1]

    # Ensure images are 3D (height, width, channels)
    if img1_np.ndim == 2:
        img1_np = np.expand_dims(img1_np, axis=-1)
    if img2_np.ndim == 2:
        img2_np = np.expand_dims(img2_np, axis=-1)

    # Calculate PSNR
    psnr_value = psnr(img1_np, img2_np, data_range=1.0)

    # Calculate SSIM
    ssim_value = ssim(img1_np, img2_np, data_range=1.0, channel_axis=-1, win_size=3)
    return psnr_value, ssim_value

def generate_image_cloud(pipeline, prompt, early_stop_threshold=2, max_steps=100):
    images_at_steps = []
    psnr_values = []
    ssim_values = []
    last_image = None
    prev_psnr = None
    early_exited_image = None
    early_exit_step = None
    start_time = time.time()

    def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
        nonlocal last_image, prev_psnr, early_exited_image, early_exit_step
        with torch.no_grad():
            image = pipeline.decode_latents(latents)
            image = pipeline.numpy_to_pil(image)[0]
            images_at_steps.append(image)

            if last_image is not None:
                psnr_value, ssim_value = image_difference(image, last_image)
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)

                if prev_psnr is not None and 5*step >=  4* max_steps:
                    psnr_diff = (prev_psnr- psnr_value)
                    if psnr_diff >= early_stop_threshold:
                        early_exited_image = image
                        early_exit_step = step

                prev_psnr = psnr_value

            last_image = image

    pipeline(prompt, num_inference_steps=max_steps, callback=callback, callback_steps=1)

    full_inference_time = time.time() - start_time
    if early_exited_image is None:
        early_exited_image=images_at_steps[-1]
        early_exit_step=max_steps


    early_exit_time=(start_time + (full_inference_time * early_exit_step / max_steps)) - start_time
    return images_at_steps, psnr_values, ssim_values, early_exited_image, early_exit_step, early_exit_time, full_inference_time

def plot_metrics(psnr_values, ssim_values, early_exit_step=None):
    steps = range(1, len(psnr_values) + 1)

    plt.figure(figsize=(12, 5))

    # PSNR plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, psnr_values, 'b-')
    plt.title('PSNR vs Inference Step')
    plt.xlabel('Inference Step')
    plt.ylabel('PSNR')
    if early_exit_step:
        plt.axvline(x=early_exit_step, color='r', linestyle='--', label='Early Exit')
        plt.legend()

    # SSIM plot
    plt.subplot(1, 2, 2)
    plt.plot(steps, ssim_values, 'r-')
    plt.title('SSIM vs Inference Step')
    plt.xlabel('Inference Step')
    plt.ylabel('SSIM')
    if early_exit_step:
        plt.axvline(x=early_exit_step, color='r', linestyle='--', label='Early Exit')
        plt.legend()

    plt.tight_layout()
    plt.show()

def run_multiple_prompts(pipeline, prompts, early_stop_threshold=2, max_steps=100):
    results = []
    for prompt in prompts:
        images, psnr_values, ssim_values, early_exited_image, early_exit_step, early_exit_time, full_inference_time = generate_image_cloud(pipeline, prompt, early_stop_threshold, max_steps)
        final_psnr, final_ssim = image_difference(images[-1], early_exited_image) if early_exited_image else (None, None)
        results.append({
            'prompt': prompt,
            'early_exit_time': early_exit_time,
            'full_inference_time': full_inference_time,
            'final_psnr': final_psnr,
            'final_ssim': final_ssim
        })
    return results

def plot_manhattan_graphs(results):
    prompts = [r['prompt'] for r in results]
    early_exit_times = [r['early_exit_time'] if r['early_exit_time'] is not None else 0 for r in results]
    full_inference_times = [r['full_inference_time'] for r in results]
    final_psnr_values = [r['final_psnr'] if r['final_psnr'] is not None else 0 for r in results]
    final_ssim_values = [r['final_ssim'] if r['final_ssim'] is not None else 0 for r in results]

    # Time comparison plot
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.step(prompts, early_exit_times, where='mid', label='Early Exit Time')
    plt.step(prompts, full_inference_times, where='mid', label='Full Inference Time')
    plt.title('Early Exit vs Full Inference Time')
    plt.xlabel('Prompts')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

    # PSNR plot
    plt.subplot(132)
    plt.step(prompts, final_psnr_values, where='mid')
    plt.title('Final PSNR Values')
    plt.xlabel('Prompts')
    plt.ylabel('PSNR')
    plt.xticks(rotation=45, ha='right')

    # SSIM plot
    plt.subplot(133)
    plt.step(prompts, final_ssim_values, where='mid')
    plt.title('Final SSIM Values')
    plt.xlabel('Prompts')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def main():
    prompts = [
          "A cute baby boy playing with his laptop",
          "A serene landscape with mountains and a lake",
          "A futuristic cityscape at night",
          "A colorful bouquet of flowers in a vase",
          "An astronaut floating in space with Earth in the background",
          "A bustling medieval market square with merchants, customers, and animals, with a castle in the background",
          "A detailed steampunk airship navigating through a sky filled with floating islands and fantastical creatures",
          "A futuristic robotic factory with various advanced machines and robots assembling parts",
          "A fantasy scene with a dragon perched on a cliff, overlooking a village in the valley below during sunset"
          # "A complex fractal art piece with vibrant colors and intricate patterns",
          # "A sunny beach with palm trees and waves gently crashing on the shore",
          # "A close-up of a butterfly resting on a flower",
          # "A classic red sports car driving on a winding road",
          # "A steaming cup of coffee on a wooden table",
          # "A cat sitting on a windowsill, looking outside"


    ]

    results = run_multiple_prompts(pipeline, prompts)

    # Plot Manhattan graphs
    plot_manhattan_graphs(results)

if __name__ == "__main__":
    main()