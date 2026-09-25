#!/usr/bin/env python3
import json
import requests
import os
from typing import Dict, Any

COMFYUI_ENDPOINT = os.getenv("COMFYUI_ENDPOINT", "http://localhost:8188")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "http://localhost:8000/runsync")

def build_workflow(prompt: str) -> Dict[str, Any]:
    """
    Build a complete ComfyUI workflow with Pony model and LoRA.
    Injects the user prompt into node 6 (positive CLIPTextEncode).
    """
    workflow = {
        "3": {
            "inputs": {
                "seed": __import__('random').randint(0, 1000000000000000),
                "steps": 25,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["10", 0],  # Model from LoraLoader
                "positive": ["6", 0],  # Positive prompt
                "negative": ["7", 0],  # Negative prompt
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 832,
                "height": 1216,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": f"{prompt}, score_9, score_8_up, score_7_up, score_6_up, masterpiece",
                "clip": ["10", 1]  # Clip from LoraLoader
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "score_1, score_2, score_3, bad anatomy, low quality",
                "clip": ["10", 1]  # Clip from LoraLoader
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        },
        "10": {
            "inputs": {
                "lora_name": "volumeloras/Uiharu_Railgun.safetensors",
                "strength_model": 0.8,
                "strength_clip": 0.8,
                "model": ["4", 0],
                "clip": ["4", 1]
            },
            "class_type": "LoraLoader"
        }
    }
    return workflow


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for RunPod API requests.
    Accepts { "input": { "prompt": "user prompt text" } }
    Returns the result from ComfyUI /runsync endpoint.
    """
    try:
        # Extract user prompt from input
        input_data = event.get("input", {})
        user_prompt = input_data.get("prompt", "1girl")  # Default fallback
        
        if not user_prompt or not isinstance(user_prompt, str):
            return {
                "error": "Missing or invalid 'prompt' parameter in input",
                "input": input_data
            }
        
        # Build the complete workflow
        workflow = build_workflow(user_prompt)
        
        # Prepare payload for ComfyUI /runsync endpoint
        payload = {
            "input": {
                "workflow": workflow
            }
        }
        
        print(f"Sending workflow to ComfyUI endpoint: {COMFYUI_ENDPOINT}/runsync")
        print(f"User prompt: {user_prompt}")
        
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
    # Test handler locally
    test_input = {
        "input": {
            "prompt": "1girl, anime, beautiful eyes"
        }
    }
    result = handler(test_input)
    print(json.dumps(result, indent=2))
