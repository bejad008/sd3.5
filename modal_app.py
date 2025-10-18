"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, Uncensored
"""

import modal
from pathlib import Path

# Inisialisasi Modal app
app = modal.App("civitai-api-fastapi")

# Default prompts untuk quality control
# VERSI BARU YANG DISARANKAN
DEFAULT_NEGATIVE_PROMPT = (
    # Kata kunci untuk mencegah konten NSFW
    "nsfw, nude, naked, porn, sex, sexual, explicit, uncensored, "
    "ass, breasts, nipple, pussy, genitalia, "
    
    # Kualitas dan anatomi (sudah bagus dari kode Anda)
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    
    # Gaya dan format yang tidak diinginkan (sudah bagus dari kode Anda)
    "(text, watermark, logo, signature), "
    "(3D, CGI, render, rendering, video game, Unreal Engine, Blender, ZBrush, painting, drawing, sketch, illustration, digital art, concept art, artwork, style, stylized, cartoon, manga, comic, 2D, flat), "
    
    # Framing (sudah bagus dari kode Anda)
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
    .pip_install(
        "fastapi[standard]",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "requests",
    )
)

# Volume untuk menyimpan model
model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
MODEL_DIR = "/models"

# Download model (jalankan sekali)
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600
)
def download_model():
    """Download model dari CivitAI"""
    import requests
    from pathlib import Path
    
    # URL Model Juggernaut XL v9
    model_url = "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16"
    model_path = Path(MODEL_DIR) / "model.safetensors"
    
    if not model_path.exists():
        print("Downloading model...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        model_volume.commit()
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")
    
    return str(model_path)


# Class untuk inference
@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    container_idle_timeout=200
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model saat container start"""
        # GANTI Impor ke versi XL
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        import torch
        
        print("Loading SDXL model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        # GANTI ke StableDiffusionXLPipeline
        self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.txt2img_pipe.to("cuda")
        
        # GANTI ke StableDiffusionXLImg2ImgPipeline
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, # SDXL pakai 2 text encoder
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, # SDXL pakai 2 tokenizer
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")
        
        print("✓ SDXL Model loaded successfully! Uncensored mode active.")
    
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
        """Generate image dari text prompt"""
        import io
        import base64
        import torch
        
        if enhance_prompt:
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        else:
            enhanced_prompt = prompt
        
        if negative_prompt is None or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        else:
            negative_prompt = str(negative_prompt).strip()
        
        print(f"Text-to-Image (SDXL): {enhanced_prompt[:100]}...")
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.txt2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
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
            "negative_prompt": negative_prompt,
            "seed": seed if seed != -1 else "random",
            "uncensored": True
        }
    
    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True
    ):
        """Edit image dengan prompt"""
        import io
        import base64
        from PIL import Image
        import torch
        
        if enhance_prompt:
            enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        else:
            enhanced_prompt = prompt
        
        if negative_prompt is None or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        else:
            negative_prompt = str(negative_prompt).strip()
        
        print(f"Image-to-Image (SDXL): {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.img2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
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
            "negative_prompt": negative_prompt,
            "strength": strength,
            "seed": seed if seed != -1 else "random",
            "uncensored": True
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
            "service": "CivitAI Model API - Uncensored (SDXL)",
            "version": "3.0",
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img"
            },
            "features": [
                "✓ No NSFW filter", 
                "✓ Uncensored generation",
                "✓ Text-to-Image (SDXL)", 
                "✓ Image-to-Image (SDXL)",
                "✓ Auto quality enhancement",
                "✓ Default negative prompts for best results"
            ],
            "default_prompts": {
                "positive_suffix": DEFAULT_POSITIVE_PROMPT_SUFFIX,
                "negative": DEFAULT_NEGATIVE_PROMPT
            }
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "service": "civitai-model-api",
            "mode": "uncensored-sdxl"
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
                "num_steps": data.get("num_steps", 25),
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
            
            api_key = data.get("custom-secret")
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
                "num_steps": data.get("num_steps", 25),
                "guidance_scale": data.get("guidance_scale", 7.5),
                "strength": data.get("strength", 0.75),
                "seed": data.get("seed", -1),
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
    """Test locally"""
    pass
