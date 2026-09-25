#!/usr/bin/env python3
import json
import os
import random
import requests
import base64
import glob
from typing import Any, Dict, List, Optional

COMFYUI_ENDPOINT = os.getenv("COMFYUI_ENDPOINT", "http://localhost:8188")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "http://localhost:8000/runsync")

DEFAULT_MODEL = os.getenv(
    "DEFAULT_MODEL",
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
)

DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "DEFAULT_NEGATIVE_PROMPT",
    "score_1, score_2, score_3, bad anatomy, low quality"
)

LORA_PATHS = [
    "/runpod-volume/loras",
    "/runpod-volume",
    "/comfyui/models/loras",
    "/comfyui/models/loras/volumeloras",
]

CHECKPOINT_PATHS = [
    "/runpod-volume/checkpoints",
    "/runpod-volume",
    "/comfyui/models/checkpoints",
]

VAE_PATHS = [
    "/comfyui/models/vae",
    "/runpod-volume/vae",
]

INPUT_IMAGE_PATH = "/comfyui/input"


def discover_loras() -> Dict[str, str]:
    loras = {}
    for base_path in LORA_PATHS:
        if not os.path.exists(base_path):
            continue
        for lora_file in glob.glob(f"{base_path}/**/*.safetensors", recursive=True):
            filename = os.path.basename(lora_file)
            loras[filename] = lora_file
            rel_path = os.path.relpath(lora_file, base_path)
            if rel_path != filename:
                loras[rel_path] = lora_file
    return loras


def discover_checkpoints() -> Dict[str, str]:
    checkpoints = {}
    for base_path in CHECKPOINT_PATHS:
        if not os.path.exists(base_path):
            continue
        for ckpt_file in glob.glob(f"{base_path}/**/*.safetensors", recursive=True):
            checkpoints[os.path.basename(ckpt_file)] = ckpt_file
    return checkpoints


def discover_vaes() -> Dict[str, str]:
    vaes = {}
    for base_path in VAE_PATHS:
        if not os.path.exists(base_path):
            continue
        for vae_file in glob.glob(f"{base_path}/**/*.safetensors", recursive=True):
            vaes[os.path.basename(vae_file)] = vae_file
    return vaes


def upload_image(image_data: str, filename: str = "input_image.png") -> str:
    os.makedirs(INPUT_IMAGE_PATH, exist_ok=True)

    if isinstance(image_data, str) and image_data.startswith("data:"):
        try:
            _, data = image_data.split(",", 1)
            image_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Invalid data URL image: {e}")
    elif isinstance(image_data, str):
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            raise ValueError("Invalid base64 image input")
    else:
        image_bytes = image_data

    filepath = os.path.join(INPUT_IMAGE_PATH, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return filename


def normalize_input(event: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(event, dict) and "input" in event and isinstance(event["input"], dict):
        return event["input"]
    if isinstance(event, dict):
        return event
    return {}


def build_workflow(
    prompt: str,
    model: str = DEFAULT_MODEL,
    loras: Optional[List[Dict[str, Any]]] = None,
    width: int = 832,
    height: int = 1216,
    steps: int = 25,
    cfg: float = 7.0,
    sampler: str = "euler",
    scheduler: str = "normal",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: Optional[int] = None,
    vae: Optional[str] = None,
    clip_skip: int = -1,
    input_image: Optional[str] = None,
    denoise: float = 1.0,
) -> Dict[str, Any]:
    if seed is None:
        seed = random.randint(0, 1000000000000000)
    if loras is None:
        loras = []

    workflow: Dict[str, Any] = {
        "4": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple"
        }
    }

    if input_image:
        workflow["1"] = {
            "inputs": {"image": input_image},
            "class_type": "LoadImage"
        }
        workflow["2"] = {
            "inputs": {
                "samples": ["1", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEEncode"
        }
        latent_source = ["2", 0]
    else:
        workflow["5"] = {
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
            "class_type": "EmptyLatentImage"
        }
        latent_source = ["5", 0]

    node_id = 10
    model_source = ["4", 0]
    clip_source = ["4", 1]

    for lora in loras:
        if not isinstance(lora, dict):
            continue
        lora_name = lora.get("name", "")
        strength_model = lora.get("strength_model", 0.8)
        strength_clip = lora.get("strength_clip", 0.8)
        if not lora_name:
            continue

        workflow[str(node_id)] = {
            "inputs": {
                "lora_name": lora_name,
                "strength_model": float(strength_model),
                "strength_clip": float(strength_clip),
                "model": model_source,
                "clip": clip_source
            },
            "class_type": "LoraLoader"
        }

        model_source = [str(node_id), 0]
        clip_source = [str(node_id), 1]
        node_id += 1

    if clip_skip != -1:
        workflow[str(node_id)] = {
            "inputs": {
                "clip": clip_source,
                "clip_skip": int(clip_skip)
            },
            "class_type": "CLIPSetLastLayer"
        }
        clip_source = [str(node_id), 0]
        node_id += 1

    workflow["6"] = {
        "inputs": {"text": prompt, "clip": clip_source},
        "class_type": "CLIPTextEncode"
    }

    workflow["7"] = {
        "inputs": {"text": negative_prompt, "clip": clip_source},
        "class_type": "CLIPTextEncode"
    }

    workflow["3"] = {
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "model": model_source,
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": latent_source
        },
        "class_type": "KSampler"
    }

    vae_source = ["4", 2]
    if vae:
        vae_node_id = node_id
        workflow[str(vae_node_id)] = {
            "inputs": {"vae_name": vae},
            "class_type": "VAELoader"
        }
        vae_source = [str(vae_node_id), 0]
        node_id += 1

    workflow["8"] = {
        "inputs": {
            "samples": ["3", 0],
            "vae": vae_source
        },
        "class_type": "VAEDecode"
    }

    workflow["9"] = {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
        },
        "class_type": "SaveImage"
    }

    return workflow


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = normalize_input(event)

        if not isinstance(input_data, dict):
            return {"error": "Missing valid input payload", "type": "invalid_input"}

        user_prompt = input_data.get("prompt", "")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return {"error": "Missing or invalid 'prompt' parameter", "input": input_data}

        model = input_data.get("model", DEFAULT_MODEL)
        loras_input = input_data.get("loras", [])
        negative_prompt = input_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        width = input_data.get("width", 832)
        height = input_data.get("height", 1216)
        steps = input_data.get("steps", 25)
        cfg = input_data.get("cfg", 7.0)
        sampler = input_data.get("sampler", "euler")
        scheduler = input_data.get("scheduler", "normal")
        vae = input_data.get("vae", None)
        clip_skip = input_data.get("clip_skip", -1)
        seed = input_data.get("seed", None)
        input_image_data = input_data.get("input_image", None)
        denoise = input_data.get("denoise", 1.0)

        discovered_loras = discover_loras()
        discovered_checkpoints = discover_checkpoints()
        discovered_vaes = discover_vaes()

        resolved_loras = []
        if isinstance(loras_input, list):
            for lora in loras_input:
                if not isinstance(lora, dict):
                    continue
                lora_name = lora.get("name", "")
                if not lora_name:
                    continue

                if lora_name in discovered_loras:
                    resolved_loras.append(lora)
                elif any(lora_name in key for key in discovered_loras.keys()):
                    matched_key = next(key for key in discovered_loras.keys() if lora_name in key)
                    new_lora = lora.copy()
                    new_lora["name"] = matched_key
                    resolved_loras.append(new_lora)
                else:
                    resolved_loras.append(lora)

        input_image_name = None
        if input_image_data:
            try:
                input_image_name = upload_image(input_image_data, "uploaded_image.png")
            except Exception as e:
                return {"error": f"Failed to upload image: {str(e)}", "type": "upload_error"}

        workflow = build_workflow(
            prompt=user_prompt,
            model=model,
            loras=resolved_loras,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            negative_prompt=negative_prompt,
            seed=seed,
            vae=vae,
            clip_skip=clip_skip,
            input_image=input_image_name,
            denoise=denoise,
        )

        payload = {"input": {"workflow": workflow}}

        response = requests.post(
            f"{COMFYUI_ENDPOINT}/runsync",
            json=payload,
            timeout=600
        )

        if response.status_code != 200:
            return {
                "error": f"ComfyUI returned status {response.status_code}",
                "details": response.text,
                "payload": payload
            }

        try:
            result = response.json()
        except ValueError:
            result = {"raw_text": response.text}

        return {
            "status": "success",
            "output": result,
            "discovered": {
                "loras": list(discovered_loras.keys()),
                "checkpoints": list(discovered_checkpoints.keys()),
                "vaes": list(discovered_vaes.keys())
            }
        }

    except requests.exceptions.RequestException as exc:
        return {"error": f"Request failed: {str(exc)}", "type": "request_error"}

    except Exception as exc:
        import traceback
        return {
            "error": f"Unexpected error: {str(exc)}",
            "type": "unknown_error",
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    examples = [
        {
            "input": {
                "prompt": "1girl, anime, beautiful eyes, masterpiece",
                "loras": [{"name": "Uiharu_Railgun.safetensors", "strength_model": 0.8, "strength_clip": 0.8}],
                "width": 832,
                "height": 1216,
                "steps": 25,
                "cfg": 7.0,
                "sampler": "euler"
            }
        },
        {
            "input": {
                "prompt": "photorealistic portrait, cinematic lighting, highly detailed",
                "negative_prompt": "low quality, blurry, bad anatomy",
                "width": 768,
                "height": 1024,
                "steps": 30,
                "cfg": 8.0,
                "sampler": "dpmpp_2m"
            }
        },
        {
            "input": {
                "prompt": "turn into anime style",
                "input_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAA...",
                "denoise": 0.7,
                "steps": 20
            }
        }
    ]

    for idx, example in enumerate(examples, start=1):
        print(f"\n=== Example {idx} ===")
        print(json.dumps(handler(example), indent=2))
