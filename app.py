import modal
import io
import base64
import os
import warnings

app = modal.App("civitai-api-fastapi")

# --- (Tidak ada perubahan di sini) ---
DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "out of frame, out of focus, cropped, "
    "(extra limbs, extra legs, extra feet, extra fingers, extra digit), "
    "(malformed hands, malformed legs, malformed feet), "
    "(missing limbs, missing legs, missing feet, missing fingers), "
    "(fused fingers, fused legs, fused feet, too many hands, bad hands, bad anatomy, double feet, kaki ganda, kaki tambahan)"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, professional photo"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "bitsandbytes",
        "sentencepiece",
        "requests", # Menambahkan 'requests' untuk download
    )
)

model_cache = modal.Volume.from_name("sdxl-juggernaut-cache-vol", create_if_missing=True)
CACHE_DIR = "/model_cache"

# --- URL Download Langsung Juggernaut v9 dari Civitai ---
JUGGERNAUT_V9_URL = "https://civitai.com/api/download/models/257749"
JUGGERNAUT_V9_FILENAME = "juggernaut-xl-v9-rundiffusion.safetensors" # Nama file bisa disesuaikan

@app.cls(
    image=image,
    gpu="L4", # Menggunakan L4
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("custom-secret")
    ],
    volumes={CACHE_DIR: model_cache},
    container_idle_timeout=300,
    timeout=1800
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        import torch
        import requests # Import untuk download
        from diffusers import StableDiffusionXLPipeline

        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # --- Logika Download dari Civitai ---
        cached_model_path = os.path.join(CACHE_DIR, JUGGERNAUT_V9_FILENAME)

        if not os.path.exists(cached_model_path):
            print(f"Model tidak ditemukan di cache. Mengunduh {JUGGERNAUT_V9_FILENAME} dari Civitai...")
            print(f"URL: {JUGGERNAUT_V9_URL}")
            
            # Download file
            with requests.get(JUGGERNAUT_V9_URL, stream=True) as r:
                r.raise_for_status() # Cek jika ada error download
                with open(cached_model_path, 'wb') as f:
                    # Download dalam chunk untuk file besar
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print("✓ Model berhasil diunduh ke cache.")
        else:
            print(f"Model {JUGGERNAUT_V9_FILENAME} ditemukan di cache.")

        # --- Memuat model dari file .safetensors tunggal ---
        print("Memuat pipeline Juggernaut-XL v9 dari file...")
        
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            cached_model_path, # Path ke file yang diunduh
            torch_dtype=torch.float16,
            use_auth_token=os.environ["HF_TOKEN"], # Tetap diperlukan untuk VAE/Text Encoder
            variant="fp16"
        )

        self.pipe.to("cuda")
        print("✓ Model Juggernaut-XL v9 berhasil dimuat!")

    @modal.method()
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 28,
        guidance_scale: float = 4.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        import torch
        
        warnings.filterwarnings("ignore", message=".*CLIP can only handle sequences up to 77 tokens.*")

        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

        print(f"Text-to-Image (Juggernaut-XL): {enhanced_prompt[:100]}...")
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None

        image = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=final_negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
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
            "seed": seed if seed != -1 else "random",
        }
    
    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 28,
        guidance_scale: float = 4.5,
        strength: float = 0.75,
        seed: int = -1, # Default -1 (random), akan di-override oleh endpoint
        enhance_prompt: bool = True
    ):
        from PIL import Image
        import torch
        
        warnings.filterwarnings("ignore", message=".*CLIP can only handle sequences up to 77 tokens.*")
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"Image-to-Image (Juggernaut-XL): {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None
        
        image = self.pipe(
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
        }

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    
    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {
            "service": "Juggernaut-XL API (from Civitai)", # Nama diperbarui
            "version": "2.1",
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img"
            },
        }

    @web_app.get("/health")
    async def health_check():
        return { "status": "healthy" }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        try:
            data = await request.json()

            if data.get("api_key") != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")

            kwargs = {
                "prompt": prompt,
                "negative_prompt": data.get("negative_prompt", ""),
                "num_steps": data.get("num_steps", 28),
                "guidance_scale": data.get("guidance_scale", 4.5),
                "width": data.get("width", 1024),
                "height": data.get("height", 1024),
                "seed": data.get("seed", -1),
                "enhance_prompt": data.get("enhance_prompt", True)
            }

            model = ModelInference()
            result = model.text_to_image.remote(**kwargs)
            return JSONResponse(content=result)

        except Exception as e:
            print(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/img2img")
    async def image_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            
            if data.get("api_key") != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image")
            if not init_image:
                raise HTTPException(status_code=400, detail="init_image is required")
            
            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "negative_prompt": data.get("negative_prompt", ""),
                "num_steps": data.get("num_steps", 28),
                "guidance_scale": data.get("guidance_scale", 4.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", 5), # Tetap menggunakan seed 5 sesuai permintaan
                "enhance_prompt": data.get("enhance_prompt", True)
            }
            
            model = ModelInference()
            result = model.image_to_image.remote(**kwargs)
            return JSONResponse(content=result)
            
        except Exception as e:
            print(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def main():
    print("Aplikasi siap untuk di-deploy ke Modal dengan perintah:")
    print("modal deploy modal_app.py")
