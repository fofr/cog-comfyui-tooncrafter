from custom_node_helper import CustomNodeHelper

MODELS = ["tooncrafter_512_interp-fp16.safetensors"]
CHECKPOINT_PATH = "ComfyUI/models/checkpoints/dynamicrafter"


class ComfyUI_DynamicCrafter(CustomNodeHelper):
    @staticmethod
    def weights_map(base_url):
        weights = {}
        for model in MODELS:
            weights[model] = {
                "url": f"{base_url}/dynamicrafter/{model}.tar",
                "dest": CHECKPOINT_PATH,
            }
        return weights
