"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, Uncensored
VERSI 4.2 - Perbaikan libGL.so.1 (OpenCV)
"""

import modal 
import io
import base64
import os
import warnings 
from pathlib import Path

# Inisialisasi Modal app
app = modal.App("civitai-api-fastapi")

# Default prompts (sudah bagus)
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, nude, naked, porn, sex, sexual, explicit, uncensored, "
    "ass, breasts, nipple, pussy, genitalia, "
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "(3D, CGI, render, rendering, video game, Unreal Engine, Blender, ZBrush, painting, drawing, sketch, illustration, digital art, concept art, artwork, style, stylized, cartoon, manga, comic, 2D, flat), "
    "out of frame, out of focus, "
    "cropped, close-up, portrait, headshot, medium shot, upper body, bust shot, face, out of frame"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, "
    "(full body shot)"
)

# Definisikan image dengan dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    
    # --- PERBAIKAN: Menambahkan dependency sistem libGL.so.1 ---
    .apt_install("libgl1-mesa-glx") 
    # -----------------------------------------------------------
    
    .pip_install(
        "fastapi[standard]",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "requests",
        "invisible-watermark", 
    )
)

# Volume untuk menyimpan model
model_volume = modal.Volume.from_name("sdxl-juggernaut-refiner-cache", create_if_missing=True)
MODEL_DIR = "/models"

# Definisikan semua URL dan Path Model
# 1. Base Model (Juggernaut v9)
BASE_MODEL_URL = "https://civitai.com/api/download/models/257749" # Ini Juggernaut v9
BASE_MODEL_FILENAME = "juggernaut-xl-v9-rundiffusion.safetensors"
BASE_MIN_SIZE_BYTES = 6_000_000_000 # 6GB

# 2. Refiner Model (Resmi SDXL)
REFINER_MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/diffusion_pytorch_model.safetensors"
REFINER_MODEL_FILENAME = "sdxl_refiner_1.0.safetensors"
REFINER_MIN_SIZE_BYTES = 5_000_000_000 # 5GB

# 3. VAE Model (Resmi SDXL)
VAE_MODEL_URL = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors"
VAE_MODEL_FILENAME = "sdxl_vae.safetensors"
VAE_MIN_SIZE_BYTES = 300_000_000 # 300MB

# Fungsi helper untuk download
def _download_file(url: str, local_path: Path, min_size: int, force: bool = False):
    """Fungsi download yang robust dengan pengecekan ukuran file."""
    import requests
    
    local_path = Path(local_path)
    
    if local_path.exists() and not force:
        try:
            file_size = local_path.stat().st_size
            if file_size >= min_size:
                print(f"✓ File valid ditemukan di cache: {local_path.name}")
                return
            else:
                print(f"File di cache korup (terlalu kecil: {file_size} bytes). Menghapus...")
                local_path.unlink()
        except Exception as e:
            print(f"Gagal mengecek file {local_path.name}: {e}. Mengunduh ulang...")
            try:
                local_path.unlink()
            except:
                pass 
    
    print(f"Mengunduh {local_path.name} dari {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        if local_path.stat().st_size < min_size:
            print(f"!!! Download Gagal. File {local_path.name} masih terlalu kecil.")
            local_path.unlink()
            raise IOError(f"Download gagal, file korup: {local_path.name}")
            
        print(f"✓ Download {local_path.name} selesai.")
        
    except Exception as e:
        print(f"!!! Gagal mengunduh {local_path.name}: {e}")
        if local_path.exists():
            local_path.unlink()
        raise

# Download model (jalankan sekali saat build)
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600
)
def download_models():
    """Download Base, Refiner, dan VAE"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    _download_file(
        BASE_MODEL_URL,
        Path(MODEL_DIR) / BASE_MODEL_FILENAME,
        BASE_MIN_SIZE_BYTES
    )
    
    _download_file(
        REFINER_MODEL_URL,
        Path(MODEL_DIR) / REFINER_MODEL_FILENAME,
        REFINER_MIN_SIZE_BYTES
    )
    
    _download_file(
        VAE_MODEL_URL,
        Path(MODEL_DIR) / VAE_MODEL_FILENAME,
        VAE_MIN_SIZE_BYTES
    )
    
    model_volume.commit()
    print("✓ Semua model (Base, Refiner, VAE) telah diunduh.")
    return True


# Class untuk inference
@app.cls(
    image=image,
    gpu="L4", 
    volumes={MODEL_DIR: model_volume},
    scaledown_window=200 
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model saat container start"""
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
        import torch
        
        base_model_path = f"{MODEL_DIR}/{BASE_MODEL_FILENAME}"
        refiner_model_path = f"{MODEL_DIR}/{REFINER_MODEL_FILENAME}"
        vae_model_path = f"{MODEL_DIR}/{VAE_MODEL_FILENAME}"

        print("Memuat VAE...")
        self.vae = AutoencoderKL.from_single_file(
            vae_model_path,
            torch_dtype=torch.float16
        )
        
        print("Memuat Base Model (Juggernaut)...")
        self.base_pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path,
            vae=self.vae, 
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.base_pipe.to("cuda")
        
        print("Memuat Refiner Model...")
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            refiner_model_path,
            vae=self.vae, 
            text_encoder_2=self.base_pipe.text_encoder_2,
            tokenizer_2=self.base_pipe.tokenizer_2,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.refiner_pipe.to("cuda")
        
        print("✓ Model Base + Refiner + VAE berhasil dimuat! Uncensored mode active.")
    
    @modal.method()
    def text_to_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        num_steps: int = 30, 
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Generate image dari text prompt menggunakan alur kerja Base + Refiner"""
        import io
        import base64
        import torch
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"Text-to-Image (Base+Refiner): {enhanced_prompt[:100]}...")
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        high_noise_frac = 0.8
        
        image_latents = self.base_pipe(
            prompt=enhanced_prompt,
            negative_prompt=final_negative_prompt,
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            output_type="latent", 
            denoising_end=high_noise_frac 
        ).images

        image = self.refiner_pipe(
            prompt=enhanced_prompt,
            negative_prompt=final_negative_prompt,
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale,
            generator=generator,
            image=image_latents, 
            denoising_start=high_noise_frac 
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": final_negative_prompt,
            "seed": seed if seed != -1 else "random",
            "uncensored": True,
            "workflow": "Base + Refiner"
        }
    
    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 30, 
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Edit image dengan prompt (Hanya menggunakan Base untuk img2img)"""
        import io
        import base64
        from PIL import Image
        import torch
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"Image-to-Image (Base Only): {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.base_pipe(
            prompt=enhanced_prompt,
            negative_prompt=final_negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": final_negative_prompt,
            "strength": strength,
            "seed": seed if seed != -1 else "random",
            "uncensored": True,
            "workflow": "Base Only" 
        }

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {
            "service": "CivitAI Model API - Uncensored (SDXL Base + Refiner)",
            "version": "4.2", # Versi diperbarui
            "gpu": "L4",
            "default_steps": 30,
            "default_i2i_seed": 5,
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img"
            },
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "service": "civitai-model-api",
            "mode": "uncensored-sdxl-base-refiner" 
        }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            
            api_key = data.get("api_key")
            expected_key = os.environ.get("API_KEY")
            
            if api_key != expected_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            kwargs = {
                "prompt": prompt,
                "num_steps": data.get("num_steps", 30), 
                "guidance_scale": data.get("guidance_scale", 7.5),
                "width": data.get("width", 1024),
                "height": data.get("height", 1024),
                "seed": data.get("seed", -1),
                "enhance_prompt": data.get("enhance_prompt", True)
            }
            
            neg = data.get("negative_prompt")
            if neg and str(neg).strip():
                kwargs["negative_prompt"] = str(neg).strip()
            
            model = ModelInference()
            result = model.text_to_image.remote(**kwargs)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/img2img")
    async def image_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            
            api_key = data.get("api_key") 
            expected_key = os.environ.get("API_KEY")
            
            if api_key != expected_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image")
            if not init_image:
                raise HTTPException(status_code=400, detail="init_image is required")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "num_steps": data.get("num_steps", 30), 
                "guidance_scale": data.get("guidance_scale", 7.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", 5), 
                "enhance_prompt": data.get("enhance_prompt", True)
            }
            
            neg = data.get("negative_prompt")
            if neg and str(neg).strip():
                kwargs["negative_prompt"] = str(neg).strip()
            
            model = ModelInference()
            result = model.image_to_image.remote(**kwargs)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app

@app.local_entrypoint()
def main():
    """
    Jalankan perintah ini di terminal Anda untuk men-deploy:
    modal deploy modal_app.py
    
    Atau untuk men-deploy builder (download model) saja:
    modal run modal_app.py::download_models
    """
    print("Menjalankan local entrypoint (tidak melakukan apa-apa).")
    print("Untuk men-deploy, jalankan: modal deploy modal_app.py")
    pass
