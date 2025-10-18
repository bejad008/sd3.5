import modal 
import io
import base64
import os
from pathlib import Path

# Ganti nama app-nya kalo lo mau, tapi ini ga wajib
app = modal.App("civitai-api-fastapi")

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    # -------------------------------------------------------------------
    # FIX 1: "liblzma5" DITAMBAHKAN DI SINI BUAT ERROR _lzma
    # -------------------------------------------------------------------
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libxext6", "libsm6", "liblzma5") 
    .pip_install(
        "fastapi[standard]",
        "torch==2.1.0",
        "diffusers==0.24.0",
        "transformers==4.35.2",
        "accelerate==0.25.0",
        "safetensors==0.4.1",
        "Pillow==10.1.0",
        "requests==2.31.0",
        "invisible-watermark==0.2.0",
    )
)

model_volume = modal.Volume.from_name("sdxl-juggernaut-refiner-cache", create_if_missing=True)
MODEL_DIR = "/models"

BASE_MODEL_URL = "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor&size=full&fp=fp16"
BASE_MODEL_FILENAME = "civitai_model.safetensors"
BASE_MIN_SIZE_BYTES = 3_000_000_000

REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
VAE_MODEL_ID = "stabilityai/sdxl-vae"

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
    
    print(f"Mengunduh {local_path.name}...")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1048576):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = (downloaded / total) * 100
                            print(f"  {pct:.1f}% ({downloaded/1e9:.2f}GB / {total/1e9:.2f}GB)")
        
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

@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
    # -------------------------------------------------------------------
    # FIX 2: SECRET HUGGING FACE DITAMBAHKAN DI SINI
    # (Nama "huggingface-secret" harus sama persis kayak di screenshot lo)
    # -------------------------------------------------------------------
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_models():
    """Download Base Model dari CivitAI"""
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionXLImg2ImgPipeline
    import torch
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MEMULAI DOWNLOAD MODEL")
    print("=" * 60)
    
    print("\n[1/2] Download Base Model (CivitAI)...")
    _download_file(
        BASE_MODEL_URL,
        Path(MODEL_DIR) / BASE_MODEL_FILENAME,
        BASE_MIN_SIZE_BYTES
    )
    
    print("\n[2/2] Preload Refiner + VAE dari HuggingFace...")
    # Token dari secret akan otomatis dipake di sini
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=MODEL_DIR
    )
    print("✓ VAE preloaded")
    
    # Token dari secret akan otomatis dipake di sini
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=MODEL_DIR
    )
    print("✓ Refiner preloaded")
    
    model_volume.commit()
    print("\n" + "=" * 60)
    print("✓ SEMUA MODEL BERHASIL DIDOWNLOAD")
    print("=" * 60 + "\n")
    return True


@app.cls(
    image=image,
    gpu="L4", 
    volumes={MODEL_DIR: model_volume},
    scaledown_window=200,
    # -------------------------------------------------------------------
    # FIX 2 (lagi): SECRET HUGGING FACE JUGA DITAMBAHKAN DI SINI
    # -------------------------------------------------------------------
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        """Load model saat container start"""
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        from diffusers.models import AutoencoderKL
        import torch
        
        base_model_path = f"{MODEL_DIR}/{BASE_MODEL_FILENAME}"
        
        print("\n" + "=" * 60)
        print("MEMUAT MODEL INFERENCE")
        print("=" * 60)
        
        try:
            print("\n[1/3] Memuat VAE...")
            # Token dari secret akan otomatis dipake di sini
            self.vae = AutoencoderKL.from_pretrained(
                VAE_MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                cache_dir=MODEL_DIR
            )
            print("✓ VAE dimuat")
            
            print("\n[2/3] Memuat Base Model (CivitAI)...")
            self.base_pipe = StableDiffusionXLPipeline.from_single_file(
                base_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                vae=self.vae
            )
            self.base_pipe.to("cuda")
            self.base_pipe.enable_attention_slicing()
            self.base_pipe.enable_vae_tiling()
            print("✓ Base Model dimuat")
            
            print("\n[3/3] Memuat Refiner Model...")
            # Token dari secret akan otomatis dipake di sini
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                REFINER_MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                cache_dir=MODEL_DIR,
                text_encoder_2=self.base_pipe.text_encoder_2,
                tokenizer_2=self.base_pipe.tokenizer_2,
                vae=self.vae
            )
            self.refiner_pipe.to("cuda")
            self.refiner_pipe.enable_attention_slicing()
            self.refiner_pipe.enable_vae_tiling()
            print("✓ Refiner Model dimuat")
            
            print("\n" + "=" * 60)
            print("✓ SEMUA MODEL BERHASIL DIMUAT - UNCENSORED MODE ACTIVE")
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"\n!!! ERROR: {str(e)}")
            print("=" * 60)
            raise
    
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
        import torch
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"\n[T2I] Prompt: {enhanced_prompt[:100]}...")
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        high_noise_frac = 0.8
        
        print(f"[Base] Generating latents (noise_frac={high_noise_frac})...")
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

        print(f"[Refiner] Refining output...")
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
        """Edit image dengan prompt"""
        from PIL import Image
        import torch
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        final_negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        
        print(f"\n[I2I] Prompt: {enhanced_prompt[:100]}...")
        
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
    # Jangan lupa lo punya secret API_KEY, tetep dipake
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
            "version": "6.0",
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
    print("Untuk men-deploy: modal deploy modal_app.py")
    print("Untuk preload model: modal run modal_app.py::download_models")
