#!/usr/bin/env python3
import json
import requests
import os
import random
from typing import Dict, Any, Optional, List

COMFYUI_ENDPOINT = os.getenv("COMFYUI_ENDPOINT", "http://localhost:8188")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "http://localhost:8000/runsync")

def build_workflow(
    prompt: str,
    model: str = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    loras: Optional[List[Dict[str, Any]]] = None,
    width: int = 832,
    height: int = 1216,
    steps: int = 25,
    cfg: float = 7.0,
    sampler: str = "euler",
    scheduler: str = "normal",
    negative_prompt: str = "score_1, score_2, score_3, bad anatomy, low quality",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build a universal ComfyUI workflow with customizable model, LoRAs, and parameters.
    
    Args:
        prompt: Main positive prompt (e.g., "1girl, anime, beautiful eyes")
        model: Checkpoint model filename (default: Pony model)
        loras: List of LoRAs to apply. Each item: {"name": "path/to/lora.safetensors", "strength_model": 0.8, "strength_clip": 0.8}
        width: Image width (default: 832)
        height: Image height (default: 1216)
        steps: Sampling steps (default: 25)
        cfg: CFG scale (default: 7.0)
        sampler: Sampler type (default: "euler")
        scheduler: Scheduler type (default: "normal")
        negative_prompt: Negative prompt
        seed: Random seed (auto-generated if None)
    
    Returns:
        ComfyUI workflow dictionary
    """
    
    if seed is None:
        seed = random.randint(0, 1000000000000000)
    
    if loras is None:
        loras = []
    
    # Start with checkpoint loader
    workflow = {
        "4": {
            "inputs": {
                "ckpt_name": model
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        }
    }
    
    # Build LoRA chain
    node_id = 10
    model_source = ["4", 0]  # Start from checkpoint
    clip_source = ["4", 1]
    
    for lora in loras:
        lora_name = lora.get("name", "")
        strength_model = lora.get("strength_model", 0.8)
        strength_clip = lora.get("strength_clip", 0.8)
        
        if not lora_name:
            continue
        
        workflow[str(node_id)] = {
            "inputs": {
                "lora_name": lora_name,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
                "model": model_source,
                "clip": clip_source
            },
            "class_type": "LoraLoader"
        }
        
        # Update sources for next LoRA or sampler
        model_source = [str(node_id), 0]
        clip_source = [str(node_id), 1]
        node_id += 1
    
    # Text encoding nodes
    workflow["6"] = {
        "inputs": {
            "text": f"{prompt}, score_9, score_8_up, score_7_up, score_6_up, masterpiece",
            "clip": clip_source
        },
        "class_type": "CLIPTextEncode"
    }
    
    workflow["7"] = {
        "inputs": {
            "text": negative_prompt,
            "clip": clip_source
        },
        "class_type": "CLIPTextEncode"
    }
    
    # KSampler
    workflow["3"] = {
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1,
            "model": model_source,
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    }
    
    # VAE Decode and Save
    workflow["8"] = {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
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
    """
    Universal RunPod handler for image generation.
    
    Input format:
    {
        "input": {
            "prompt": "1girl, anime, beautiful eyes",
            "model": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",  # Optional
            "loras": [
                {"name": "volumeloras/Uiharu_Railgun.safetensors", "strength_model": 0.8, "strength_clip": 0.8},
                {"name": "volumeloras/another_lora.safetensors", "strength_model": 0.6, "strength_clip": 0.6}
            ],  # Optional
            "width": 832,  # Optional
            "height": 1216,  # Optional
            "steps": 25,  # Optional
            "cfg": 7.0,  # Optional
            "sampler": "euler",  # Optional
            "scheduler": "normal",  # Optional
            "negative_prompt": "score_1, score_2, score_3, bad anatomy, low quality",  # Optional
            "seed": 12345  # Optional
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        # Extract required prompt
        user_prompt = input_data.get("prompt", "")
        if not user_prompt or not isinstance(user_prompt, str):
            return {
                "error": "Missing or invalid 'prompt' parameter in input",
                "input": input_data
            }
        
        # Extract optional parameters with defaults
        model = input_data.get("model", "ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
        loras = input_data.get("loras", [])
        width = input_data.get("width", 832)
        height = input_data.get("height", 1216)
        steps = input_data.get("steps", 25)
        cfg = input_data.get("cfg", 7.0)
        sampler = input_data.get("sampler", "euler")
        scheduler = input_data.get("scheduler", "normal")
        negative_prompt = input_data.get("negative_prompt", "score_1, score_2, score_3, bad anatomy, low quality")
        seed = input_data.get("seed", None)
        
        # Validate parameters
        if not isinstance(loras, list):
            loras = []
        
        # Build the workflow
        workflow = build_workflow(
            prompt=user_prompt,
            model=model,
            loras=loras,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            negative_prompt=negative_prompt,
            seed=seed
        )
        
        # Prepare payload
        payload = {
            "input": {
                "workflow": workflow
            }
        }
        
        print(f"Sending workflow to ComfyUI endpoint: {COMFYUI_ENDPOINT}/runsync")
        print(f"Prompt: {user_prompt}")
        print(f"Model: {model}")
        print(f"LoRAs: {len(loras)} loaded")
        print(f"Resolution: {width}x{height}, Steps: {steps}, CFG: {cfg}")
        
        # Send to ComfyUI
        response = requests.post(
            f"{COMFYUI_ENDPOINT}/runsync",
            json=payload,
            timeout=600  # 10 minute timeout for generation
        )
        
        # Check for errors
        if response.status_code != 200:
            return {
                "error": f"ComfyUI returned status {response.status_code}",
                "details": response.text
            }
        
        result = response.json()
        
        # Success
        return {
            "status": "success",
            "output": result
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request failed: {str(e)}",
            "type": "request_error"
        }
    
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parsing failed: {str(e)}",
            "type": "json_error"
        }
    
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "type": "unknown_error"
        }


if __name__ == "__main__":
    # Test 1: Simple prompt with default Pony model and single LoRA
    test_input_1 = {
        "input": {
            "prompt": "1girl, anime, beautiful eyes",
            "loras": [
                {"name": "volumeloras/Uiharu_Railgun.safetensors", "strength_model": 0.8, "strength_clip": 0.8}
            ]
        }
    }
    print("Test 1: Single LoRA")
    print(json.dumps(handler(test_input_1), indent=2))
    print("\n" + "="*80 + "\n")
    
    # Test 2: Multiple LoRAs
    test_input_2 = {
        "input": {
            "prompt": "2girls, kissing, anime style",
            "loras": [
                {"name": "volumeloras/Uiharu_Railgun.safetensors", "strength_model": 0.7, "strength_clip": 0.7},
                {"name": "volumeloras/another_character.safetensors", "strength_model": 0.6, "strength_clip": 0.6}
            ],
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg": 8.0
        }
    }
    print("Test 2: Multiple LoRAs with custom settings")
    print(json.dumps(handler(test_input_2), indent=2))
    print("\n" + "="*80 + "\n")
    
    # Test 3: Different model, no LoRAs
    test_input_3 = {
        "input": {
            "prompt": "landscape, beautiful sunset",
            "model": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
            "width": 1280,
            "height": 720
        }
    }
    print("Test 3: Different resolution, no LoRAs")
    print(json.dumps(handler(test_input_3), indent=2))
