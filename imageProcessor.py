import torch
from collections import Counter
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import os
import random
import cv2
from rembg import remove

import numpy as np
import requests

# import spacy
from openai import OpenAI
import json
import re
from fpdf import FPDF

# from google.colab import auth
import math

# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload


class ImageProcessor:
    def __init__(self):
        self.caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.nlp = spacy.load("en_core_web_sm")

        self.client = OpenAI(
            api_key="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"
        )

    def load_image(self, image_path, target_width=1024):
        """Loads an image, resizes it to the target width while maintaining aspect ratio, and converts to RGB."""
        img = Image.open(image_path).convert("RGB")  # Resize while maintaining quality
        return img

    def create_curvy_header(
        self,
        secondary_color,
        primary_color,
        output_path="header.png",
        width=595,
        height=120,
    ):
        """Generates a smooth curvy header image using Matplotlib with a primary color background."""

        # Define x values (from 0 to width)
        x = np.linspace(0, width, 100)

        # Define a smooth sine wave for the curve
        y = height - (
            30 + 20 * np.sin(2 * np.pi * x / width * 3)
        )  # Adjust frequency for waves

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Set the background color to primary_color
        fig.patch.set_facecolor(np.array(primary_color) / 255.0)
        ax.set_facecolor(np.array(primary_color) / 255.0)

        # Fill the area below the curve with the secondary color
        ax.fill_between(x, y, height, color=np.array(secondary_color) / 255.0)

        # Remove axes
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis("off")

        # Save as a PNG with the primary color background
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

    def create_pdf_with_image(
        final_image_path, image_summary, secondary_color, dominant_color, output_pdf
    ):
        """Creates a PDF with a curvy header, sharp image, centered image, and visible text."""

        pdf = FPDF(unit="pt", format="A4")  # Use points for precise positioning
        pdf.add_page()

        # Insert a curvy header image (ensure this exists)
        header_image_path = "header.png"

        pdf.image(header_image_path, x=0, y=0, w=595)  # Full width of A4

        # Insert flipped header as footer
        flipped_header_path = "flipped_header.png"
        img = Image.open(header_image_path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        img.save(flipped_header_path)

        pdf.image(flipped_header_path, x=0, y=725, w=595)  # Footer position

        # Set title text in the header
        pdf.set_font("Arial", style="B", size=24)
        pdf.set_text_color(255, 255, 255)  # White text
        pdf.set_xy(200, 40)  # Adjust title position
        pdf.cell(200, 30, "Title", align="C")

        # Set background color for the rest of the page
        pdf.set_fill_color(dominant_color[0], dominant_color[1], dominant_color[2])
        pdf.rect(0, 100, 595, 652, "F")  # Background below the header

        # Load and enhance the image
        img = Image.open(final_image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(4.0)  # Increase sharpness

        # Resize image (scaled to 1/3 of original size)
        original_width, original_height = img.size
        new_width = original_width // 3
        new_height = original_height // 3
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save resized image temporarily
        resized_img_path = "temp_resized.png"
        img.save(resized_img_path)

        # Center the image
        image_x = (595 - new_width) // 2  # Centered horizontally
        pdf.image(resized_img_path, x=image_x, y=150, w=new_width, h=new_height)

        # Add image summary below the image
        pdf.set_xy(50, 150 + new_height + 40)  # Position below image
        pdf.set_font("Arial", style="B", size=18)
        pdf.set_text_color(255, 255, 255)  # White text
        pdf.multi_cell(495, 50, f"{image_summary}", align="C")

        # Save the final PDF
        pdf.output(output_pdf)

        # Cleanup temp images
        os.remove(resized_img_path)
        os.remove(flipped_header_path)

        print(f"PDF saved at {output_pdf}")

    def add_text_to_image(self, image_path, text, image_summary, output_path):
        """Adds bold text with a white rounded rectangle background, padding, and better readability."""

        # Open image and convert to RGBA
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size

        # Choose font style dynamically
        font_style = self.get_best_font_style(image_summary)

        font_paths = {
            "futuristic": "Wide awake Black.ttf",
            "calligraphy": "GOUDY.TTF",
            "normal": "arialbd.ttf",
        }

        try:
            font = ImageFont.truetype(font_paths[font_style], 14)
        except IOError:
            font = ImageFont.load_default()

        # Measure text dimensions
        draw = ImageDraw.Draw(img)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Set up padding and spacing
        text_padding = 15  # Space inside the text box
        text_bg_width = text_width + (2 * text_padding)
        text_bg_height = text_height + (2 * text_padding)

        # Create a new transparent image
        new_width = width + text_bg_width + 20  # Add space for the text box
        new_height = max(height, text_bg_height)
        new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Paste the original image
        new_img.paste(img, (0, (new_height - height) // 2), img)

        # Text box position
        text_x = width + 10
        text_y = (new_height - text_bg_height) // 2

        # Draw rounded rectangle background
        draw = ImageDraw.Draw(new_img)
        draw.rounded_rectangle(
            [text_x, text_y, text_x + text_bg_width, text_y + text_bg_height],
            fill="white",
            radius=30,  # Smoother rounded edges
        )

        # Correct text positioning inside the box
        draw.text(
            (text_x + text_padding, text_y + text_padding),
            text,
            fill="black",
            font=font,
        )

        # Save output
        new_img.save(output_path)
        return output_path

    def image_summary(self, image_path):
        image = self.load_image(image_path)
        inputs = self.caption_processor(image, return_tensors="pt")
        out = self.caption_model.generate(**inputs)
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def refine_prompt(self, image_summary):
        """Uses GPT-4o to generate an optimized prompt for icon generation."""
        chat_prompt = (
            f"Extract the most relevant subject from the following description: `{image_summary}`. "
            f"Return a **single word or simple phrase** best representing the subject for a **minimalist, "
            f"monochrome, line-art SVG icon**. Example: If given 'pencil box,' return 'pencil'; "
            f"if given 'water bottle,' return 'water droplet'. Keep it simple and clear."
        )

        try:
            chat_response = self.client.chat.completions.create(
                model="gpt-4o",  # Correct model syntax
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.3,  # Keep the output consistent
                max_tokens=10,  # Ensure a short response
            )
            print(chat_response.choices[0].message.content.strip())
            return chat_response.choices[
                0
            ].message.content.strip()  # Correct way to access response

        except Exception as e:
            print(f"Failed to refine prompt: {e}")
            return "default icon"

    def get_product_advantages(self, image_summary):
        """Generates 5 key advantages of the product based on its description."""

        chat_prompt = (
            f"Provide **exactly 4 key advantages** each of max 3 words, of this product : `{image_summary}` "
            f"in valid JSON format as an array **without markdown formatting**. "
            f'Example: ["Advantage 1", "Advantage 2", "Advantage 3", "Advantage 4", "Advantage 5"]'
        )

        try:
            chat_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            # Get response text
            response_text = chat_response.choices[0].message.content.strip()
            print(f"Raw API Response: {response_text}")  # Debugging output

            # Remove markdown JSON formatting if present (e.g., ```json ... ```)
            cleaned_response = re.sub(
                r"```json\s*([\s\S]*?)\s*```", r"\1", response_text
            ).strip()

            # Parse the cleaned JSON string safely
            advantages = json.loads(cleaned_response)

            if isinstance(advantages, list) and len(advantages) == 4:
                return advantages
            else:
                raise ValueError("Invalid response format")

        except Exception as e:
            print(f"Failed to get advantages: {e}")
            return []

    def get_best_font_style(self, image_summary):
        """Determines the best font type (futuristic, calligraphy, or normal) based on the product description."""

        chat_prompt = (
            f"Based on this product description: `{image_summary}`, determine the best suitable font type. "
            f'Choose only one word from: "futuristic", "calligraphy", or "normal". '
            f"Respond with only the single word without markdown formatting."
        )

        try:
            chat_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.3,
                max_tokens=10,
            )

            # Get response text
            response_text = chat_response.choices[0].message.content.strip()
            print(f"Raw API Response: {response_text}")  # Debugging output

            # Clean the response (if any unwanted formatting exists)
            cleaned_response = re.sub(r"[^\w]", "", response_text).lower()

            # Ensure response is one of the three expected values
            valid_fonts = {"futuristic", "calligraphy", "normal"}
            if cleaned_response in valid_fonts:
                return cleaned_response
            else:
                raise ValueError("Invalid font response received")

        except Exception as e:
            print(f"Error in get_best_font_style: {e}")
            return "normal"  # Default fallback

    def generate_icon(
        self,
        image_summary,
        product_final,
        save_path="icon.png",
    ):
        """Generates an icon based on an optimized prompt from ChatGPT."""
        refined_subject = self.refine_prompt(image_summary)  # Get optimized subject
        product_final.append(refined_subject)

        image_prompt = (
            f"Minimalistic, white background black icon, line-art SVG icon, with only one subject of : '{refined_subject}', clean and simple. "
            f"Ensure clarity and universal recognizability."
        )

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            img_data = requests.get(image_url).content

            with open(save_path, "wb") as f:
                f.write(img_data)

            print(f"Icon saved as {save_path}")
            return save_path

        except Exception as e:
            print(f"Failed to generate icon: {e}")
            return "default.svg"

    def image_similarity(self, image_paths):
        images = [self.load_image(path) for path in image_paths]
        inputs = self.clip_processor(images=images, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)
        similarity = torch.cosine_similarity(outputs[0], outputs[1], dim=0).item()
        return similarity

    def create_geometric_background(
        self, image_size, primary_color, secondary_color, pattern_type
    ):
        width, height = image_size
        bg = Image.new("RGB", image_size, primary_color)
        draw = ImageDraw.Draw(bg)

        if pattern_type == "stripes":
            for i in range(0, width, 40):
                draw.rectangle([i, 0, i + 20, height], fill=secondary_color)

        elif pattern_type == "circles":
            circles = []
            for _ in range(80):
                attempts = 0
                while attempts < 10:
                    x, y = random.randint(0, width), random.randint(0, height)
                    radius = random.randint(5, 15)
                    overlap = any(
                        ((x - cx) ** 2 + (y - cy) ** 2) < ((radius + cr + 5) ** 2)
                        for cx, cy, cr in circles
                    )
                    if not overlap:
                        circles.append((x, y, radius))
                        draw.ellipse(
                            [x - radius, y - radius, x + radius, y + radius],
                            fill=secondary_color,
                        )
                        break
                    attempts += 1

        elif pattern_type == "grid":
            for x in range(0, width, 50):
                for y in range(0, height, 50):
                    draw.rectangle([x, y, x + 25, y + 25], fill=secondary_color)

        elif pattern_type == "triangles":
            triangle_size = 50  # Adjust size of triangles
            triangles = []  # Store drawn triangle positions

            for x in range(0, width, triangle_size):
                for y in range(0, height, triangle_size):
                    # Ensure all triangles point upwards (△ shape)
                    x1, y1 = x, y + triangle_size
                    x2, y2 = x + triangle_size, y + triangle_size
                    x3, y3 = x + triangle_size // 2, y  # Always at the top

                    # Check if the new triangle overlaps an existing one
                    overlap = any(
                        (
                            abs(x1 - tx1) < triangle_size
                            and abs(y1 - ty1) < triangle_size
                        )
                        for tx1, ty1, _, _, _, _ in triangles
                    )

                    if not overlap:
                        triangles.append((x1, y1, x2, y2, x3, y3))
                        draw.polygon([x1, y1, x2, y2, x3, y3], fill=secondary_color)
        elif pattern_type == "hexagons":
            hex_size = 40
            for row in range(0, height, int(hex_size * 1.5)):
                for col in range(0, width, int(hex_size * 2)):
                    x_offset = hex_size if (row // int(hex_size * 1.5)) % 2 else 0
                    x, y = col + x_offset, row
                    hex_points = [
                        (x, y),
                        (x + hex_size, y - hex_size // 2),
                        (x + hex_size * 2, y),
                        (x + hex_size * 2, y + hex_size),
                        (x + hex_size, y + hex_size * 1.5),
                        (x, y + hex_size),
                    ]
                    draw.polygon(hex_points, fill=secondary_color)

        elif pattern_type == "diagonal":
            for i in range(-width, width, 40):
                draw.line([i, 0, i + 40, height], fill=secondary_color, width=10)

        elif pattern_type == "waves":
            for x in range(0, width, 60):
                for y in range(0, height, 60):
                    draw.arc(
                        [x, y, x + 60, y + 60],
                        start=0,
                        end=180,
                        fill=secondary_color,
                        width=6,
                    )
        center_x, center_y = width // 2, height // 2
        center_radius = min(width, height) // 3
        center_color = (
            (255, 255, 255) if primary_color != (255, 255, 255) else (0, 0, 0)
        )
        draw.ellipse(
            [
                center_x - center_radius,
                center_y - center_radius,
                center_x + center_radius,
                center_y + center_radius,
            ],
            fill=center_color,
        )

        border_thickness = 5
        if pattern_type == "stripes" or pattern_type == "diagonal":
            for i in range(0, width, 20):
                draw.line(
                    [i, 0, i, border_thickness],
                    fill=secondary_color,
                    width=border_thickness,
                )
                draw.line(
                    [i, height - border_thickness, i, height],
                    fill=secondary_color,
                    width=border_thickness,
                )
        elif pattern_type == "circles":
            draw.ellipse(
                [
                    border_thickness,
                    border_thickness,
                    width - border_thickness,
                    height - border_thickness,
                ],
                outline=secondary_color,
                width=border_thickness,
            )
        elif pattern_type == "grid":
            for x in range(0, width, 50):
                draw.rectangle([x, 0, x + 25, border_thickness], fill=secondary_color)
                draw.rectangle(
                    [x, height - border_thickness, x + 25, height], fill=secondary_color
                )
        elif pattern_type == "triangles":
            for i in range(0, width, 50):
                draw.polygon(
                    [(i, 0), (i + 50, border_thickness), (i + 50, 0)],
                    fill=secondary_color,
                )
                draw.polygon(
                    [
                        (i, height),
                        (i + 50, height - border_thickness),
                        (i + 50, height),
                    ],
                    fill=secondary_color,
                )
        elif pattern_type == "hexagons":
            hex_size = 40
            for i in range(0, width, int(hex_size * 2)):
                draw.polygon(
                    [(i, 0), (i + hex_size, border_thickness), (i + hex_size * 2, 0)],
                    fill=secondary_color,
                )
                draw.polygon(
                    [
                        (i, height),
                        (i + hex_size, height - border_thickness),
                        (i + hex_size * 2, height),
                    ],
                    fill=secondary_color,
                )

        return bg

    def add_object_border(self, image_path, output_path="output.png", border_size=30):
        image = Image.open(image_path).convert("RGBA")
        removed_bg = remove(image)
        image_array = np.array(removed_bg)
        alpha_channel = image_array[:, :, 3]
        blurred = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        border_mask = np.zeros_like(binary_mask)
        cv2.drawContours(border_mask, contours, -1, (255), border_size)
        border_only = cv2.subtract(border_mask, binary_mask)
        border_image = np.zeros_like(image_array)
        border_image[:, :, :3] = (0, 0, 0)  # Black border
        border_image[:, :, 3] = border_only  # Keep transparency for everything el
        border_pil = Image.fromarray(border_image)
        final_image = Image.alpha_composite(border_pil, removed_bg)
        final_image.save(output_path)
        return output_path

    def merge_images(self, subject_path, background, output_path):
        subject = Image.open(subject_path).convert("RGBA")

        # Scale down the subject to 1/2 its original size
        width, height = subject.size
        subject = subject.resize((width // 2, height // 2), Image.LANCZOS)

        # Ensure background is RGBA
        background = background.convert("RGBA")

        # Compute the position to center the subject on the background
        bg_width, bg_height = background.size
        pos_x = (bg_width - subject.width) // 2
        pos_y = (bg_height - subject.height) // 2

        # Paste the subject onto the background
        merged = background.copy()
        merged.paste(subject, (pos_x, pos_y), subject)

        # Save the merged image
        merged.save(output_path)
        return output_path

    def get_dominant_color(self, image_path):
        """Extracts the dominant color while avoiding black/white."""
        image = Image.open(image_path).convert("RGB")
        pixels = list(image.getdata())

        # Filter out white and black
        filtered_pixels = [p for p in pixels if sum(p) > 60 and sum(p) < 700]
        if not filtered_pixels:
            return (0, 0, 200), (100, 100, 255)  # Default to blue

        color_counter = Counter(filtered_pixels)
        for _ in range(10):
            most_common = color_counter.most_common(1)
            if not most_common:
                break
            dominant_color = most_common[0][0]
            if sum(dominant_color) < 60 or sum(dominant_color) > 700:
                color_counter.pop(dominant_color, None)
                continue

            # Generate darker & lighter variations
            darker = tuple(max(0, c - 40) for c in dominant_color)
            lighter = tuple(min(255, c + 40) for c in dominant_color)
            return darker, lighter

        return (0, 0, 200), (100, 100, 255)  # Default blue

    def remove_background(self, image_path):
        image = self.load_image(image_path)
        output = remove(image).convert("RGBA")
        output_path = image_path
        output.save(output_path)
        return output_path

    def create_circular_image(
        self, subject_path, image_path, output_path="circular_icon.png"
    ):
        """Creates a 50px x 50px circular image with a solid background (no transparency)."""

        size = (50, 50)  # Fixed size for the circular image
        image = Image.open(image_path).convert("RGBA")
        image = image.resize(size, Image.LANCZOS)  # Resize to match the circular area

        _, bg_color = self.get_dominant_color(subject_path)

        # **Create a base image with a solid background**
        circular_image = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(circular_image)
        draw.ellipse((0, 0, 50, 50), fill=bg_color)  # Draw solid circular background

        # **Create a circular mask for transparency**
        mask = Image.new("L", size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, 50, 50), fill=255)  # Circular mask

        # **Paste the resized image onto the circle**
        paste_x = (size[0] - image.width) // 2
        paste_y = (size[1] - image.height) // 2
        circular_image.paste(image, (paste_x, paste_y), image)

        # **Apply the circular mask**
        circular_image = Image.composite(
            circular_image, Image.new("RGBA", size, bg_color), mask
        )

        # **Convert to RGB to remove transparency but keep circular shape**
        final_image = Image.new("RGB", size, bg_color)  # Solid background
        final_image.paste(
            circular_image, (0, 0), mask
        )  # Apply the mask to retain circle

        # **Save the final circular image**
        final_image.save(output_path)
        return output_path

    def generate_and_merge_images(
        self,
        image_paths,
        subject_path,
        background_path,
        final_image_path,
        product_final,
    ):
        """
        Merges a background, centers a scaled-down subject, and places four scaled-up icons at the corners with margins.
        """

        # drive_service = build('drive', 'v3')
        # Open main images
        subject = Image.open(subject_path).convert("RGBA")
        background = Image.open(background_path).convert("RGBA")
        final_image = background.copy()
        bg_width, bg_height = background.size

        # Resize subject to have a fixed width of 1024px while maintaining aspect ratio
        subj_width, subj_height = subject.size
        new_width = 1024
        aspect_ratio = subj_height / subj_width
        new_height = int(new_width * aspect_ratio)  # Maintain aspect ratio

        subject = subject.resize((new_width, new_height), Image.LANCZOS)

        # Center the subject
        pos_x = (bg_width - new_width) // 2
        pos_y = (bg_height - new_height) // 2
        final_image.paste(subject, (pos_x, pos_y), subject)

        # Scale icons by 2.3x
        scale_factor = 2.3
        margin = 60
        scaled_icons = []

        # Place four corner icons with scaling
        for path in image_paths[:4]:
            icon = Image.open(path).convert("RGBA")
            width, height = icon.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled_icon = icon.resize((new_width, new_height), Image.LANCZOS)
            scaled_icons.append(scaled_icon)

        # Define corner positions dynamically based on scaled sizes
        corners = [
            (margin, margin),
            (bg_width - new_width - margin, margin),
            (margin, bg_height - new_height - margin),
            (bg_width - new_width - margin, bg_height - new_height - margin),
        ]

        for icon, (x, y) in zip(scaled_icons, corners):
            final_image.paste(icon, (x, y), icon)

        # Save final image
        final_image.save(final_image_path)
        print(f"Final image saved at: {final_image_path}")

        return final_image_path
