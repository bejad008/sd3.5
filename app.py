import modal
import io
import base64
import os
from typing import Optional
import gc

app = modal.App("civitai-api-fastapi")

# ============================================================================
# CONSTANTS
# ============================================================================

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

# ============================================================================
# DOCKER IMAGE
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]==0.115.5",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "Pillow==11.0.0",
        "sentencepiece==0.2.0",
    )
)

model_cache = modal.Volume.from_name("image-gen-cache", create_if_missing=True)
CACHE_DIR = "/model_cache"

# ============================================================================
# STABLE DIFFUSION 3.5 CLASS (Text-to-Image)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={CACHE_DIR: model_cache},
    dcaledown_window=300,
    timeout=900,
    # KUNCI: Allow 10 concurrent requests per container
    allow_concurrent_inputs=10,
    # Modal akan auto-scale hingga max containers
)
class StableDiffusion35:
    """
    Dedicated container untuk Stable Diffusion 3.5 Large (Text-to-Image only)
    Optimized untuk handle multiple concurrent requests
    """
    
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import StableDiffusion3Pipeline
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        print("=" * 70)
        print("🚀 Loading Stable Diffusion 3.5 Large...")
        print("=" * 70)
        
        model_id = "stabilityai/stable-diffusion-3.5-large"
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=os.environ.get("HF_TOKEN"),
            cache_dir=CACHE_DIR
        )
        
        # LANGSUNG KE GPU (karena container ini dedicated untuk SD3 only)
        self.pipe.to("cuda")
        
        # Memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing(1)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        # Enable xformers jika tersedia (untuk speed boost)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled")
        except:
            print("⚠️  xFormers not available")
        
        print("✅ SD3.5 ready on GPU!")
        print("🔥 Can handle 10 concurrent requests per container")
        print("=" * 70)
    
    def _validate_dimensions(self, width: int, height: int) -> tuple:
        """Validasi dan koreksi dimensi gambar"""
        width = max(MIN_IMAGE_SIZE, min(width, MAX_IMAGE_SIZE))
        height = max(MIN_IMAGE_SIZE, min(height, MAX_IMAGE_SIZE))
        # Kelipatan 8
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
            # Validasi
            width, height = self._validate_dimensions(width, height)
            num_steps = self._validate_steps(num_steps)
            
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
            final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
            
            print(f"🎨 [SD3] Generating: {enhanced_prompt[:60]}... ({width}x{height})")
            
            generator = None
            if seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(seed)
            
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
            
            # Cleanup
            del generator
            torch.cuda.empty_cache()
            gc.collect()
            
            # Encode
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            print(f"✅ [SD3] Done!")
            
            return {
                "success": True,
                "image": img_str,
                "prompt": enhanced_prompt,
                "original_prompt": prompt,
                "negative_prompt": final_negative_prompt,
                "seed": seed if seed != -1 else "random",
                "width": width,
                "height": height,
                "steps": num_steps,
                "model": "SD3.5-Large"
            }
            
        except Exception as e:
            print(f"❌ [SD3] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "model": "SD3.5-Large"
            }

# ============================================================================
# QWEN IMAGE EDIT CLASS (Image-to-Image)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={CACHE_DIR: model_cache},
    container_idle_timeout=300,
    timeout=900,
    # KUNCI: Allow 10 concurrent requests per container
    allow_concurrent_inputs=10,
)
class QwenImageEdit:
    """
    Dedicated container untuk Qwen Image Edit (Image-to-Image only)
    Optimized untuk handle multiple concurrent requests
    """
    
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import AutoPipelineForImage2Image
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        print("=" * 70)
        print("🚀 Loading Qwen Image Edit...")
        print("=" * 70)
        
        model_id = "Qwen/Qwen-Image-Edit"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
        
        # LANGSUNG KE GPU (karena container ini dedicated untuk Qwen only)
        self.pipe.to("cuda")
        
        # Memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing(1)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled")
        except:
            print("⚠️  xFormers not available")
        
        print("✅ Qwen ready on GPU!")
        print("🔥 Can handle 10 concurrent requests per container")
        print("=" * 70)
    
    def _validate_steps(self, steps: int) -> int:
        return max(MIN_STEPS, min(steps, MAX_STEPS))

    @modal.method()
    def edit(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1
    ):
        from PIL import Image
        import torch
        
        try:
            num_steps = self._validate_steps(num_steps)
            strength = max(0.1, min(1.0, strength))
            
            print(f"🖼️  [Qwen] Editing: {prompt[:60]}...")
            
            # Decode image
            try:
                init_image_bytes = base64.b64decode(init_image_b64)
                init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image data: {e}")
            
            # Resize jika terlalu besar
            w, h = init_image.size
            if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
                ratio = min(MAX_IMAGE_SIZE/w, MAX_IMAGE_SIZE/h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
                print(f"📏 Resized: {w}x{h} → {new_w}x{new_h}")
            
            generator = None
            if seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(seed)
            
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    image=init_image,
                    negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator
                )
                image = result.images[0]
            
            # Cleanup
            del generator
            torch.cuda.empty_cache()
            gc.collect()
            
            # Encode
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            print(f"✅ [Qwen] Done!")
            
            return {
                "success": True,
                "image": img_str,
                "prompt": prompt,
                "negative_prompt": negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT,
                "strength": strength,
                "seed": seed if seed != -1 else "random",
                "steps": num_steps,
                "model": "Qwen-Image-Edit"
            }
            
        except Exception as e:
            print(f"❌ [Qwen] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "model": "Qwen-Image-Edit"
            }

# ============================================================================
# FASTAPI GATEWAY (Lightweight, no GPU)
# ============================================================================

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("custom-secret")],
    # Gateway tidak butuh GPU, bisa scale banyak
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import time
    
    web_app = FastAPI(
        title="Telegram Bot Image Generation API",
        version="4.0-PRODUCTION-SCALE",
        description="Handles 1000+ concurrent requests with auto-scaling"
    )
    
    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request metrics (untuk monitoring)
    request_count = {"text2img": 0, "img2img": 0}
    
    def increment_counter(endpoint: str):
        request_count[endpoint] += 1

    @web_app.get("/")
    async def root():
        return {
            "service": "Telegram Bot Image Generation API",
            "version": "4.0-PRODUCTION-SCALE",
            "status": "operational",
            "architecture": {
                "description": "Separated model containers with auto-scaling",
                "text2img_model": "SD3.5-Large (dedicated GPU containers)",
                "img2img_model": "Qwen-Image-Edit (dedicated GPU containers)",
                "concurrency_per_container": 10,
                "auto_scaling": "enabled",
                "max_queue_size": "unlimited (Modal managed)"
            },
            "endpoints": {
                "health": "GET /health",
                "metrics": "GET /metrics",
                "text2img": "POST /text2img",
                "img2img": "POST /img2img"
            },
            "auth": "X-API-Key header required"
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": time.time()
        }
    
    @web_app.get("/metrics")
    async def metrics(x_api_key: Optional[str] = Header(None)):
        # Simple metrics endpoint (bisa dipake untuk monitoring)
        if x_api_key != os.environ.get("API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return {
            "total_requests": request_count,
            "timestamp": time.time()
        }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(
        request: Request,
        background_tasks: BackgroundTasks,
        x_api_key: Optional[str] = Header(None)
    ):
        """
        Text-to-Image endpoint menggunakan SD3.5 Large
        Auto-scales berdasarkan load
        """
        try:
            data = await request.json()
            api_key = x_api_key or data.get("api_key")
            
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            prompt = data.get("prompt", "").strip()
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt required")
            
            if len(prompt) > 1000:
                raise HTTPException(status_code=400, detail="Prompt too long (max 1000)")

            # Track metrics di background
            background_tasks.add_task(increment_counter, "text2img")

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

            # KUNCI: Modal auto-scale containers berdasarkan queue
            # Jika ada 1000 request, Modal akan spin up multiple containers
            # Setiap container handle 10 concurrent requests
            model = StableDiffusion35()
            result = model.generate.remote(**kwargs)
            
            if not result.get("success"):
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return JSONResponse(content=result)

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Gateway error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/img2img")
    async def image_to_image_endpoint(
        request: Request,
        background_tasks: BackgroundTasks,
        x_api_key: Optional[str] = Header(None)
    ):
        """
        Image-to-Image endpoint menggunakan Qwen Image Edit
        Auto-scales berdasarkan load
        """
        try:
            data = await request.json()
            api_key = x_api_key or data.get("api_key")
            
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image", "").strip()
            if not init_image:
                raise HTTPException(status_code=400, detail="init_image required")
            
            prompt = data.get("prompt", "").strip()
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt required")
            
            if len(prompt) > 1000:
                raise HTTPException(status_code=400, detail="Prompt too long")
            
            background_tasks.add_task(increment_counter, "img2img")
            
            kwargs = {
                "init_image_b64": init_image,
                "prompt": prompt,
                "negative_prompt": data.get("negative_prompt", ""),
                "num_steps": data.get("num_steps", 50),
                "guidance_scale": data.get("guidance_scale", 7.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", -1)
            }
            
            model = QwenImageEdit()
            result = model.edit.remote(**kwargs)
            
            if not result.get("success"):
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Gateway error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    print("=" * 80)
    print("🚀 TELEGRAM BOT IMAGE GENERATION API - PRODUCTION SCALE")
    print("=" * 80)
    print("\n📊 Architecture:")
    print("   • Separated model containers (SD3.5 + Qwen)")
    print("   • 10 concurrent requests per container")
    print("   • Auto-scaling based on queue length")
    print("   • Lightweight FastAPI gateway")
    print("\n📈 Capacity:")
    print("   • Target: 1000+ concurrent requests")
    print("   • Modal will auto-scale containers to meet demand")
    print("   • Requests queue automatically when all containers busy")
    print("\n🔧 Commands:")
    print("   modal deploy app.py          # Deploy to production")
    print("   modal serve app.py           # Local development")
    print("\n📊 Monitoring:")
    print("   modal app logs telegram-bot-image-api")
    print("   GET /metrics                 # Request statistics")
    print("=" * 80)
