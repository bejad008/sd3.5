"""
Deploy Stable Diffusion 3.5 Large ke Modal.com dengan FastAPI
Metode ini menggunakan from_pretrained untuk memuat model dari Hugging Face.
"""

import modal
import io
import base64
import os

# --- Konfigurasi App ---
app = modal.App("civitai-api-fastapi")

# --- Default Prompts ---
DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "out of frame, out of focus, cropped, "
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, professional photo"
)

# --- Definisi Image Container ---
# Menambahkan 'bitsandbytes' untuk efisiensi
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
    )
)

# --- Class untuk Inference Model ---
@app.cls(
    image=image,
    gpu="L4",  # T4 atau yang lebih baik sangat direkomendasikan
    # Rahasia untuk otentikasi Hugging Face dan API Key Anda
    secrets=[
        modal.Secret.from_name("huggingface-secret"), # WAJIB untuk download model
        modal.Secret.from_name("custom-secret")      # Untuk API Key endpoint Anda
    ],
    container_idle_timeout=300,
    timeout=1800 # Timeout lebih lama untuk download model pertama kali
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """
        Load model saat container start.
        Fungsi from_pretrained akan men-download dan men-cache model secara otomatis.
        """
        import torch
        from diffusers import StableDiffusion3Pipeline

        print("Memuat model Stable Diffusion 3.5 Large...")
        model_id = "stabilityai/stable-diffusion-3.5-large"

        # Menggunakan .from_pretrained dengan token dari Modal Secrets
        # Ini menggantikan fungsi download_model() dan from_single_file()
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=os.environ["HF_TOKEN"]
        )

        # Kirim model ke GPU
        self.pipe.to("cuda")
        print("✓ Model SD 3.5 Large berhasil dimuat!")

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
        """Generate gambar dari teks"""
        import torch

        # Logika prompt tetap sama seperti kode original Anda
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

        print(f"Text-to-Image (SD3.5): {enhanced_prompt[:100]}...")
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
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Edit gambar dengan prompt"""
        from PIL import Image
        import torch
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"Image-to-Image (SD3.5): {enhanced_prompt[:100]}...")
        
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


# --- Aplikasi FastAPI ---
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
            "service": "Stable Diffusion 3.5 Large API",
            "version": "1.1",
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

            # Otentikasi API Key
            if data.get("api_key") != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            prompt = data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required")

            # Mengambil parameter dari request, dengan default value
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
                raise HTTPException(status_code=400, detail="prompt is required")
            
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "negative_prompt": data.get("negative_prompt", ""),
                "num_steps": data.get("num_steps", 28),
                "guidance_scale": data.get("guidance_scale", 4.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", -1),
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
    """Fungsi ini hanya berjalan jika Anda menjalankan skrip secara lokal."""
    print("Aplikasi siap untuk di-deploy ke Modal dengan perintah:")
    print("modal deploy modal_app.py")
