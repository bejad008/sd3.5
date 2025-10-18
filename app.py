import modal
import io
import base64
import os
from pathlib import Path

app = modal.App("civitai-api-fastapi") # <-- Nama app lo

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchvision",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "sentencepiece",
    )
)

model_cache = modal.Volume.from_name("sd3-model-cache-vol", create_if_missing=True)
CACHE_DIR = "/model_cache"

@app.cls(
    image=image,
    gpu="A100", # Tetap pake A100
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
        from diffusers import StableDiffusion3Pipeline, QwenImageEditPipeline

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.device = "cuda"
        
        # --- INI FIX-NYA ---
        print("Memuat model Stable Diffusion 3.5 Large...")
        model_id_sd3 = "stabilityai/stable-diffusion-3.5-large"
        self.sd3_pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id_sd3,
            torch_dtype=torch.float16,
            use_auth_token=os.environ["HF_TOKEN"],
            cache_dir=CACHE_DIR
        )
        # LANGSUNG PAKSA PINDAH KE CPU BIAR VRAM KOSONG
        self.sd3_pipe.to("cpu") 
        print("✓ Model SD 3.5 Large berhasil dimuat (di CPU).")

        print("Memuat model Qwen Image Edit...")
        model_id_qwen = "Qwen/Qwen-Image-Edit" 
        self.qwen_pipe = QwenImageEditPipeline.from_pretrained(
            model_id_qwen,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            use_auth_token=os.environ["HF_TOKEN"],
        )
        # LANGSUNG PAKSA PINDAH KE CPU BIAR VRAM KOSONG
        self.qwen_pipe.to("cpu")
        print("✓ Model Qwen Image Edit berhasil dimuat (di CPU).")
        print("--- VRAM KOSONG, SIAP MENERIMA REQUEST ---")


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
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

        print(f"Text-to-Image (SD3.5): {enhanced_prompt[:100]}...")
        
        # --- Sistem Ganti Jaga ---
        print("Memindahkan SD3.5 ke GPU...")
        self.sd3_pipe.to(self.device)
        print("✓ SD3.5 di GPU.")
        
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

        image = self.sd3_pipe(
            prompt=enhanced_prompt,
            negative_prompt=final_negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        # --- Sistem Ganti Jaga ---
        print("Memindahkan SD3.5 kembali ke CPU...")
        self.sd3_pipe.to("cpu")
        print("✓ SD3.5 di CPU.")

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
        num_steps: int = 50,
        guidance_scale: float = 4.0,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        from PIL import Image
        import torch
        
        print(f"Image-to-Image (Qwen): {prompt[:100]}...")
        
        # --- Sistem Ganti Jaga ---
        print("Memindahkan Qwen ke GPU...")
        self.qwen_pipe.to(self.device)
        print("✓ Qwen di GPU.")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
        
        image = self.qwen_pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or " ", 
            generator=generator,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_steps
        ).images[0]
        
        # --- Sistem Ganti Jaga ---
        print("Memindahkan Qwen kembali ke CPU...")
        self.qwen_pipe.to("cpu")
        print("✓ Qwen di CPU.")
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt": prompt,
            "original_prompt": prompt,
            "negative_prompt": negative_prompt.strip() or " ",
            "strength": "N/A (Qwen Model)",
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
            "service": "Multi-Model API",
            "version": "2.2-OOM-FIX (SD3.5 + Qwen-Edit)",
            "gpu": "A100", # Info GPU
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img (Stable Diffusion 3.5)",
                "image-to-image": "POST /img2img (Qwen Image Edit)"
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
                raise HTTPException(status_code=400, detail="prompt is required (ini adalah instruksi edit)")
            
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "negative_prompt": data.get("negative_prompt", ""),
                "num_steps": data.get("num_steps", 50),
                "guidance_scale": data.get("guidance_scale", 4.0),
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
    print("Aplikasi siap untuk di-deploy ke Modal dengan perintah:")
    print("modal deploy app.py")
