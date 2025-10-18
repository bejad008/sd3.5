import modal
import io
import base64
import os
from pathlib import Path
from typing import Optional
import gc

app = modal.App("civitai-api-fastapi")  # TETAP ORIGINAL

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

MAX_IMAGE_SIZE = 2048
MIN_IMAGE_SIZE = 256
MAX_STEPS = 100
MIN_STEPS = 1

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

# ============================================================================
# OPTIMASI: Pisahkan SD3.5 ke container sendiri (TETAP MODEL ORIGINAL LO)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={CACHE_DIR: model_cache},
    scaledown_window=300,
    timeout=1800
)
@modal.concurrent(10)  # OPTIMASI: 10 concurrent per container
class SD35Model:
    """Dedicated container untuk SD3.5 - LANGSUNG DI GPU, NO SWITCHING"""
    
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import StableDiffusion3Pipeline

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.device = "cuda"
        
        print("Memuat model Stable Diffusion 3.5 Large...")
        model_id_sd3 = "stabilityai/stable-diffusion-3.5-large"  # TETAP MODEL LO
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id_sd3,
            torch_dtype=torch.float16,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=CACHE_DIR
        )
        
        # OPTIMASI: Langsung ke GPU, no CPU switching
        self.pipe.to(self.device)
        
        # Memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing(1)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled")
        except:
            pass
        
        print("✓ Model SD 3.5 Large berhasil dimuat (DI GPU - OPTIMIZED)")
    
    def _validate_dimensions(self, width: int, height: int) -> tuple:
        width = max(MIN_IMAGE_SIZE, min(width, MAX_IMAGE_SIZE))
        height = max(MIN_IMAGE_SIZE, min(height, MAX_IMAGE_SIZE))
        width = (width // 8) * 8
        height = (height // 8) * 8
        return width, height
    
    def _validate_steps(self, steps: int) -> int:
        return max(MIN_STEPS, min(steps, MAX_STEPS))

    @modal.method()
    def generate(
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
        
        try:
            width, height = self._validate_dimensions(width, height)
            num_steps = self._validate_steps(num_steps)
            
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
            final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

            print(f"Text-to-Image (SD3.5): {enhanced_prompt[:100]}...")
            
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

            # OPTIMASI: No CPU/GPU switching, langsung inference
            with torch.inference_mode():
                image = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=final_negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
            
            # Cleanup after inference
            del generator
            torch.cuda.empty_cache()
            gc.collect()

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return {
                "success": True,
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": prompt,
                "negative_prompt": final_negative_prompt,
                "seed": seed if seed != -1 else "random",
            }
        except Exception as e:
            print(f"Error in text_to_image: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# OPTIMASI: Pisahkan Qwen ke container sendiri (TETAP MODEL ORIGINAL LO)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={CACHE_DIR: model_cache},
    scaledown_window=300,
    timeout=1800
)
@modal.concurrent(10)  # OPTIMASI: 10 concurrent per container
class QwenModel:
    """Dedicated container untuk Qwen - LANGSUNG DI GPU, NO SWITCHING"""
    
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import AutoPipelineForImage2Image

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.device = "cuda"
        
        print("Memuat model Qwen Image Edit...")
        model_id_qwen = "Qwen/Qwen-Image-Edit"  # TETAP MODEL LO
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id_qwen,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        )
        
        # OPTIMASI: Langsung ke GPU, no CPU switching
        self.pipe.to(self.device)
        
        # Memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing(1)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled")
        except:
            pass
        
        print("✓ Model Qwen Image Edit berhasil dimuat (DI GPU - OPTIMIZED)")
    
    def _validate_steps(self, steps: int) -> int:
        return max(MIN_STEPS, min(steps, MAX_STEPS))
    
    @modal.method()
    def edit(
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
        
        try:
            num_steps = self._validate_steps(num_steps)
            
            print(f"Image-to-Image (Qwen): {prompt[:100]}...")
            
            init_image_bytes = base64.b64decode(init_image_b64)
            init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
            
            # Resize jika terlalu besar
            w, h = init_image.size
            if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
                ratio = min(MAX_IMAGE_SIZE/w, MAX_IMAGE_SIZE/h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
                print(f"Resized: {w}x{h} -> {new_w}x{new_h}")
            
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
            
            # OPTIMASI: No CPU/GPU switching, langsung inference
            with torch.inference_mode():
                image = self.pipe(
                    image=init_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    strength=strength
                ).images[0]
            
            # Cleanup after inference
            del generator
            torch.cuda.empty_cache()
            gc.collect()
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "image": img_str,
                "prompt": prompt,
                "original_prompt": prompt,
                "negative_prompt": negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT,
                "strength": strength,
                "seed": seed if seed != -1 else "random",
            }
        except Exception as e:
            print(f"Error in image_to_image: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# FASTAPI - TETAP ENDPOINT ORIGINAL LO: /text2img dan /img2img
# ============================================================================

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")]
)
@modal.concurrent(100)  # OPTIMASI: Gateway bisa handle 100 concurrent
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI()
    
    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/")
    async def root():
        return {
            "service": "Multi-Model API",
            "version": "3.0-OPTIMIZED (SD3.5 + Qwen-Edit)",
            "gpu": "A100",
            "architecture": "Separated containers with auto-scaling",
            "concurrency_per_container": 10,
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img (Stable Diffusion 3.5)",
                "image-to-image": "POST /img2img (Qwen Image Edit)"
            },
        }

    @web_app.get("/health")
    async def health_check():
        return { "status": "healthy" }

    @web_app.post("/text2img")  # TETAP ENDPOINT LO
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

            # OPTIMASI: Panggil SD3 dedicated container
            model = SD35Model()
            result = model.generate.remote(**kwargs)
            return JSONResponse(content=result)

        except Exception as e:
            print(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/img2img")  # TETAP ENDPOINT LO
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
            
            # OPTIMASI: Panggil Qwen dedicated container
            model = QwenModel()
            result = model.edit.remote(**kwargs)
            return JSONResponse(content=result)
            
        except Exception as e:
            print(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def main():
    print("=" * 80)
    print("Aplikasi OPTIMIZED siap untuk di-deploy ke Modal")
    print("=" * 80)
    print("\nOPTIMASI yang diterapkan:")
    print("✅ SD3.5 dan Qwen di container terpisah (NO CPU/GPU switching overhead)")
    print("✅ Setiap container handle 10 concurrent requests")
    print("✅ Modal auto-scale containers sesuai load")
    print("✅ Gateway bisa handle 100 concurrent")
    print("\nYang TIDAK diubah:")
    print("✅ App name: civitai-api-fastapi")
    print("✅ Model: SD3.5 + Qwen (original)")
    print("✅ Endpoint: /text2img dan /img2img (original)")
    print("\nDeploy:")
    print("   modal deploy app.py")
    print("=" * 80)
