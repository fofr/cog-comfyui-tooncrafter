# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
        check_orientation: bool = True,
    ):
        image = Image.open(input_file)

        if check_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

        image.save(os.path.join(INPUT_DIR, filename))

    def handle_images_connections(self, workflow, total_images, is_loop):
        batch_images = workflow["28"]["inputs"]
        batch_images["inputcount"] = total_images + 1 if is_loop else total_images

        for index in range(1, total_images + 1):
            if index > 2:
                load_image = {
                    "inputs": {"image": f"input_{index}.png", "upload": "image"},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Image"},
                }

                load_image_index = f"{index+300}"
                batch_images_input = [f"{load_image_index}", 0]
                batch_images[f"image_{index}"] = batch_images_input
                workflow[load_image_index] = load_image

        if is_loop:
            batch_images[f"image_{total_images+1}"] = ["1", 0]

    def update_workflow(self, workflow, **kwargs):
        self.handle_images_connections(workflow, kwargs["total_images"], kwargs["loop"])

        positive_prompt = workflow["49"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["50"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        image_resize = workflow["70"]["inputs"]
        image_resize["width"] = kwargs["max_width"]
        image_resize["height"] = kwargs["max_height"]

        toon_crafter_interpolation = workflow["57"]["inputs"]
        toon_crafter_interpolation["seed"] = kwargs["seed"]

        video_helper = workflow["29"]["inputs"]
        video_upscaler = workflow["73"]["inputs"]

        if not kwargs["color_correction"]:
            del workflow["67"]
            video_helper["images"] = ["58", 0]
            video_upscaler["images"] = ["58", 0]

        if not kwargs["interpolate"]:
            del workflow["71"]
            del workflow["72"]
            del workflow["73"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        max_width: int = Input(
            description="Width of the image",
            default=512,
            ge=256,
            le=768,
        ),
        max_height: int = Input(
            description="Height of the image",
            default=512,
            ge=256,
            le=768,
        ),
        image_1: Path = Input(
            description="First input image",
        ),
        image_2: Path = Input(
            description="Second input image",
        ),
        image_3: Path = Input(
            description="Third input image (optional)",
            default=None,
        ),
        image_4: Path = Input(
            description="Fourth input image (optional)",
            default=None,
        ),
        image_5: Path = Input(
            description="Fifth input image (optional)",
            default=None,
        ),
        image_6: Path = Input(
            description="Sixth input image (optional)",
            default=None,
        ),
        image_7: Path = Input(
            description="Seventh input image (optional)",
            default=None,
        ),
        image_8: Path = Input(
            description="Eighth input image (optional)",
            default=None,
        ),
        image_9: Path = Input(
            description="Ninth input image (optional)",
            default=None,
        ),
        image_10: Path = Input(
            description="Tenth input image (optional)",
            default=None,
        ),
        loop: bool = Input(
            description="Loop the video",
            default=False,
        ),
        interpolate: bool = Input(
            description="Interpolate video",
            default=False,
        ),
        color_correction: bool = Input(
            description="Color correction",
            default=True,
        ),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        images = [
            img
            for img in [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                image_7,
                image_8,
                image_9,
                image_10,
            ]
            if img is not None
        ]

        for index, image in enumerate(images, start=1):
            self.handle_input_file(image, filename=f"input_{index}.png")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            loop=loop,
            seed=seed,
            total_images=len(images),
            interpolate=interpolate,
            color_correction=color_correction,
            max_width=max_width,
            max_height=max_height,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])
