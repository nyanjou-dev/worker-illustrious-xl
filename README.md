# worker-illustrious-xl

A [RunPod](https://runpod.io) serverless worker for **Illustrious XL v1.0** — a high-quality anime-focused SDXL-based model by [Liberata](https://huggingface.co/Liberata/illustrious-xl-v1.0).

Built on top of the [worker-sdxl](https://github.com/runpod-workers/worker-sdxl) template, adapted to use Illustrious XL instead of the base SDXL pipeline. The refiner stage has been removed — Illustrious XL produces high-quality results in a single pass.

---

## API

### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | — | Generation prompt |
| `negative_prompt` | string | `null` | Negative prompt |
| `width` | int | `1024` | Output width in pixels |
| `height` | int | `1024` | Output height in pixels |
| `num_inference_steps` | int | `28` | Number of denoising steps |
| `guidance_scale` | float | `7.0` | CFG scale |
| `scheduler` | string | `K_EULER_ANCESTRAL` | Scheduler (see below) |
| `seed` | int | random | RNG seed |
| `num_images` | int | `1` | Number of images (max 2) |

**Supported schedulers:** `PNDM`, `KLMS`, `DDIM`, `K_EULER`, `K_EULER_ANCESTRAL`, `DPMSolverMultistep`, `DPMSolverSinglestep`

### Output

```json
{
  "images": ["data:image/png;base64,..."],
  "image_url": "data:image/png;base64,...",
  "seed": 42
}
```

If `BUCKET_ENDPOINT_URL` is set in the environment, images are uploaded to S3-compatible storage and URLs are returned instead of base64.

---

## Example Input

```json
{
  "input": {
    "prompt": "1girl, white hair, fox ears, shrine maiden, detailed face, masterpiece, best quality",
    "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature, bad anatomy, worst quality",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 7.0,
    "scheduler": "K_EULER_ANCESTRAL",
    "seed": 42
  }
}
```

---

## Docker Build

```bash
docker build --build-arg HF_TOKEN=<your_hf_token> -t worker-illustrious-xl .
```

The `HF_TOKEN` build arg is passed through for model download. If the model is public, it can be omitted.

## Local Testing (RunPod)

```bash
docker run --gpus all -e RUNPOD_WEBHOOK_GET_JOB="" worker-illustrious-xl
```

Or with [runpod-python](https://github.com/runpod/runpod-python) test mode:

```bash
python handler.py --test_input test_input.json
```

---

## Model

- **Model**: [Liberata/illustrious-xl-v1.0](https://huggingface.co/Liberata/illustrious-xl-v1.0)
- **Base arch**: Stable Diffusion XL
- **Pipeline**: `StableDiffusionXLPipeline` (diffusers)
- **No refiner** — single-pass generation

---

## License

See [LICENSE](LICENSE).
