from flask import Flask, request, jsonify, send_from_directory, send_file
from pymongo import MongoClient
import pymongo.errors
import os
from flask_cors import CORS
from imageProcessor import ImageProcessor  # Assuming this is in image_processor.py

# Assuming these are in utils.py
import requests
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:8080",
                "https://fd-c.vercel.app",
                "http://localhost:3000",
            ]
        }
    },
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

# MongoDB Connection
mongo_uri = "mongodb+srv://Khelan05:KrxRwjRwkhgYUdwh@cluster0.c6y9phd.mongodb.net/fd1?retryWrites=true&w=majority"
try:
    client = MongoClient(mongo_uri)
    db = client["fd"]
    images_collection = db["images"]  # Collection for storing image processing results
    print("Connected to MongoDB successfully!")
except pymongo.errors.ConnectionError as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Initialize ImageProcessor
image_processor = ImageProcessor()


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        # Receive JSON data from the request
        data = request.get_json()
        image_url = data["imageUrl"]
        description = data["description"]

        # Download the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image"}), 400

        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image_path = "temp_image.png"
        image.save(image_path)

        # Create directory for merged images
        os.makedirs("mergedImages", exist_ok=True)
        product_final = []

        # Extract subject and color
        possible_colors = image_processor.get_dominant_color(image_path)
        image_summary = image_processor.image_summary(image_path)
        image_processor.generate_icon(image_summary, product_final)
        subject_path = image_processor.add_object_border(image_path)
        product_final.append(image_summary)

        # Process patterns and generate icons
        patterns = ["circles"]
        created_backgrounds = {}
        descriptions2 = image_processor.get_product_advantages(image_summary)
        icon_paths = []

        # Remove background from the generated icon and create a circular version
        removedBgIconImage = image_processor.remove_background("./icon.png")
        circularIconImage = image_processor.create_circular_image(
            image_path, removedBgIconImage
        )

        # Overlay text on the circular icon for each description
        for desc in descriptions2:
            icon_with_text = image_processor.add_text_to_image(
                circularIconImage,
                desc,
                image_summary,
                f"icon_{desc.replace(' ', '_').lower()}.png",
            )
            icon_paths.append(icon_with_text)

        # Process each pattern
        for pattern in patterns:
            primary_color = possible_colors[0]
            secondary_color = possible_colors[1]

            # Create or reuse geometric background
            bg_key = f"{primary_color}_{secondary_color}_{pattern}"
            if bg_key not in created_backgrounds:
                background = image_processor.create_geometric_background(
                    (1500, 1500), primary_color, secondary_color, pattern
                )
                bg_path = f"mergedImages/background_{pattern}.png"
                background.save(bg_path)
                created_backgrounds[bg_key] = bg_path
            else:
                bg_path = created_backgrounds[bg_key]

            # Merge subject, background, and icons
            final_image_path = f"mergedImages/final_{pattern}.png"
            image_processor.generate_and_merge_images(
                icon_paths[:4], subject_path, bg_path, final_image_path, product_final
            )

            # Generate PDF (optional, but kept as part of the process)
            output_pdf = f"mergedImages/final_{pattern}.pdf"
            image_processor.create_curvy_header(secondary_color, primary_color)
            #image_processor.create_pdf_with_image(
            #    final_image_path,
            #    image_summary,
            #    secondary_color,
            #    primary_color,
            #    output_pdf,
            #)

            # Send the generated image as a file response
            return send_file(
                final_image_path,
                mimetype="image/png",
                as_attachment=True,
                download_name="processed_image.png"
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pdfs/<filename>", methods=["GET"])
def serve_pdf(filename):
    """Serve the generated PDF from the mergedImages directory."""
    return send_from_directory("mergedImages", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
