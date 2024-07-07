

# Requirements: pip install torch transformers diffusers  accelerate  matplotlib scikit-image

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


import time
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    min_steps = int(0.8 * max_steps)  

    early_exit_time = None  # Initialize early_exit_time

    def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
        nonlocal last_image, prev_psnr, early_exited_image, early_exit_step, early_exit_time
        with torch.no_grad():
            image = pipeline.decode_latents(latents)
            image = pipeline.numpy_to_pil(image)[0]
            images_at_steps.append(image)

            if last_image is not None:
                psnr_value, ssim_value = image_difference(image, last_image)
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                # print(f"Step: {step}, PSNR: {psnr_value:.4f}")
                if prev_psnr is not None:
                    psnr_diff = (prev_psnr- psnr_value)
                    # Check for early exit condition
                    if step >= min_steps and psnr_diff >= early_stop_threshold:
                        early_exited_image = image
                        early_exit_step = step
                        early_exit_time = time.time() - start_time 

                prev_psnr = psnr_value

            last_image = image

    pipeline(prompt, num_inference_steps=max_steps, callback=callback, callback_steps=1)

    full_inference_time = time.time() - start_time
    if early_exited_image is None:
        early_exited_image = last_image
        early_exit_step = max_steps
        early_exit_time = full_inference_time

    return images_at_steps, psnr_values, ssim_values, early_exited_image, early_exit_step, early_exit_time, full_inference_time

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

import matplotlib.pyplot as plt
import numpy as np

def plot_manhattan_graphs(results, prompt_categories):
    category_results = {category: {'early_exit_times': [], 'full_inference_times': [], 'psnr_ratios': []} for category in prompt_categories}

    for result in results:
        prompt = result['prompt']
        early_exit_time = result['early_exit_time'] if result['early_exit_time'] is not None else 0
        full_inference_time = result['full_inference_time']
        final_psnr = result['final_psnr'] if result['final_psnr'] is not None else 0

        for category, prompts in prompt_categories.items():
            if prompt in prompts:
                category_results[category]['early_exit_times'].append(early_exit_time)
                category_results[category]['full_inference_times'].append(full_inference_time)
                category_results[category]['psnr_ratios'].append(final_psnr)

    # Calculate averages
    avg_early_exit_times = {category: np.mean(data['early_exit_times']) for category, data in category_results.items()}
    avg_full_inference_times = {category: np.mean(data['full_inference_times']) for category, data in category_results.items()}
    avg_psnr_ratios = {category: np.mean(data['psnr_ratios']) for category, data in category_results.items()}

    categories = list(category_results.keys())
    num_categories = len(categories)
    bar_width = 0.35
    index = np.arange(num_categories)

    # Plotting styles
    plt.style.use('seaborn-darkgrid')
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)

    # Plot Early Exit vs Full Inference Time
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(index, list(avg_early_exit_times.values()), bar_width, label='Early Exit Time', color='skyblue', alpha=0.7)
    ax.bar(index + bar_width, list(avg_full_inference_times.values()), bar_width, label='Full Inference Time', color='salmon', alpha=0.7)
    ax.set_title('Average Early Exit vs Full Inference Time', fontsize=16)
    ax.set_xlabel('Prompt Categories', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('avg_early_exit_vs_full_inference_time.png')
    plt.close()

    # Plot PSNR Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(categories, list(avg_psnr_ratios.values()), color='green', alpha=0.7)
    ax.set_title('Average PSNR Ratio', fontsize=16)
    ax.set_xlabel('Prompt Categories', fontsize=14)
    ax.set_ylabel('PSNR', fontsize=14)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('avg_psnr_ratio.png')
    plt.close()


def main():
    prompt_categories = {
        "Human Activities and Figures": [
            "A cute baby boy playing with his laptop",
            "A bustling medieval market square with merchants, customers, and animals, with a castle in the background",
            "An astronaut floating in space with Earth in the background",
            "A family having a picnic in a park",
            "A firefighter rescuing a kitten from a tree",
            "A group of friends hiking in the mountains",
            "A chef cooking in a busy kitchen",
            "A teacher giving a lecture in a classroom",
            "A musician playing a guitar at a concert",
            "A couple dancing under the stars"
        ],
        "Landscapes and Scenery": [
            "A serene landscape with mountains and a lake",
            "A sunny beach with palm trees and waves gently crashing on the shore",
            "A desert landscape with sand dunes and a clear blue sky",
            "A snowy forest with tall pine trees",
            "A tropical rainforest with a waterfall",
            "A sunset over a peaceful meadow",
            "A misty morning in a dense forest",
            "A vibrant field of sunflowers",
            "A calm river flowing through a canyon",
            "A scenic view of the Grand Canyon"
        ],
        "Futuristic and Fantasy": [
            "A futuristic cityscape at night",
            "A detailed steampunk airship navigating through a sky filled with floating islands and fantastical creatures",
            "A futuristic robotic factory with various advanced machines and robots assembling parts",
            "A fantasy scene with a dragon perched on a cliff, overlooking a village in the valley below during sunset",
            "A cyborg in a high-tech laboratory",
            "A space station orbiting a distant planet",
            "A magical forest with glowing plants and mythical creatures",
            "A wizard casting a spell in a mystical cave",
            "A spaceship landing on an alien planet",
            "A post-apocalyptic city in ruins"
        ],
        "Objects and Still Life": [
            "A colorful bouquet of flowers in a vase",
            "A steaming cup of coffee on a wooden table",
            "A classic red sports car driving on a winding road",
            "A vintage typewriter on an old wooden desk",
            "A basket of fresh fruits on a kitchen counter",
            "A pair of glasses and an open book on a bedside table",
            "A collection of antique clocks",
            "A stack of neatly folded clothes",
            "A luxurious wristwatch on a marble surface",
            "A set of artist's brushes and paints"
        ],
        "Animals and Nature": [
            "A close-up of a butterfly resting on a flower",
            "A cat sitting on a windowsill, looking outside",
            "A lioness with her cubs in the savannah",
            "A school of colorful fish swimming in a coral reef",
            "A panda eating bamboo in a bamboo forest",
            "A dog playing fetch in a park",
            "A flock of birds flying at sunset",
            "A horse galloping in an open field",
            "A squirrel gathering nuts in a forest",
            "A whale breaching the ocean surface"
        ],
        "Abstract and Complex Patterns": [
            "A complex fractal art piece with vibrant colors and intricate patterns",
            "A mandala design with intricate patterns and vibrant colors",
            "An abstract painting with bold geometric shapes and contrasting colors",
            "A mosaic made of colorful glass pieces",
            "A kaleidoscopic pattern with symmetrical shapes",
            "A digital art piece with glitch effects",
            "A 3D rendering of a geometric sculpture",
            "An abstract wave pattern with gradient colors",
            "A detailed zentangle pattern",
            "A psychedelic art piece with swirling colors"
        ]
    }

    prompts = [prompt for category in prompt_categories.values() for prompt in category]

    results = run_multiple_prompts(pipeline, prompts)

    # Plotting  graphs
    plot_manhattan_graphs(results, prompt_categories)

if __name__ == "__main__":
    main()

def plot_manhattan_graphs(results, prompt_categories):
    category_results = {category: {'early_exit_times': [], 'full_inference_times': [], 'psnr_ratios': []} for category in prompt_categories}

    for result in results:
        prompt = result['prompt']
        early_exit_time = result['early_exit_time'] if result['early_exit_time'] is not None else 0
        full_inference_time = result['full_inference_time']
        final_psnr = result['final_psnr'] if result['final_psnr'] is not None else 0

        for category, prompts in prompt_categories.items():
            if prompt in prompts:
                category_results[category]['early_exit_times'].append(early_exit_time)
                category_results[category]['full_inference_times'].append(full_inference_time)
                category_results[category]['psnr_ratios'].append(final_psnr)

    # Calculate averages
    avg_early_exit_times = {category: np.mean(data['early_exit_times']) for category, data in category_results.items()}
    avg_full_inference_times = {category: np.mean(data['full_inference_times']) for category, data in category_results.items()}
    avg_psnr_ratios = {category: np.mean(data['psnr_ratios']) for category, data in category_results.items()}

    categories = list(category_results.keys())
    num_categories = len(categories)
    bar_width = 0.35
    index = np.arange(num_categories)

    # Plotting styles
    plt.style.use('seaborn-darkgrid')
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)

    # Plot Early Exit vs Full Inference Time
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(index, list(avg_early_exit_times.values()), bar_width, label='Early Exit Time', color='skyblue', alpha=0.7)
    ax.bar(index + bar_width, list(avg_full_inference_times.values()), bar_width, label='Full Inference Time', color='salmon', alpha=0.7)
    ax.set_title('Average Early Exit vs Full Inference Time', fontsize=16)
    ax.set_xlabel('Prompt Categories', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    # Adjust y-axis limits and tick spacing
    max_time = max(max(avg_early_exit_times.values()), max(avg_full_inference_times.values()))
    ax.set_ylim(0, max_time + 10)  # Adjust as needed for better visibility
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Adjust the number of ticks as needed

    plt.tight_layout()
    plt.savefig('avg_early_exit_vs_full_inference_time.png')
    plt.close()

    # Plot PSNR Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(categories, list(avg_psnr_ratios.values()), color='green', alpha=0.7)
    ax.set_title('Average PSNR Ratio', fontsize=16)
    ax.set_xlabel('Prompt Categories', fontsize=14)
    ax.set_ylabel('PSNR', fontsize=14)
    ax.set_xticklabels(categories, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('avg_psnr_ratio.png')
    plt.close()