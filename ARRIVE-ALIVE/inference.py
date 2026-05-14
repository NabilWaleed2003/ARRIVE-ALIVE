"""Inference script for testing predictions on images or directories."""

import argparse
import torch
from pathlib import Path
from PIL import Image
from src.data.transforms import get_eval_transform
from src.utils import MODEL_SAVE_PATH, DEVICE, CLASS_NAMES


def load_model(model_path=MODEL_SAVE_PATH):
    """Load TorchScript model."""
    model = torch.jit.load(model_path)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_image(image_path, model, transform):
    """Predict drowsiness for a single image."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction_idx = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction_idx].item()
    
    return {
        'image': image_path,
        'prediction': CLASS_NAMES[prediction_idx],
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[0]: float(probabilities[0][0].item()),
            CLASS_NAMES[1]: float(probabilities[0][1].item()),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test drowsiness detection on images"
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to image file or directory containing images'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_SAVE_PATH,
        help='Path to model file'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        model = load_model(args.model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Get transform
    transform = get_eval_transform()
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        try:
            result = predict_image(input_path, model, transform)
            print(f"\nPrediction for {result['image']}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        except Exception as e:
            print(f"✗ Error processing image: {e}")
    
    elif input_path.is_dir():
        # Directory of images
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        if not image_files:
            print(f"No images found in {input_path}")
            return
        
        print(f"\nProcessing {len(image_files)} images...\n")
        
        results = []
        for img_path in image_files:
            try:
                result = predict_image(img_path, model, transform)
                results.append(result)
                print(f"{img_path.name}: {result['prediction']} ({result['confidence']:.4f})")
            except Exception as e:
                print(f"{img_path.name}: Error - {e}")
        
        # Summary
        print(f"\n{'='*60}")
        drowsy_count = sum(1 for r in results if r['prediction'] == 'Drowsy')
        non_drowsy_count = len(results) - drowsy_count
        
        print(f"Summary:")
        print(f"  Total: {len(results)}")
        print(f"  Drowsy: {drowsy_count}")
        print(f"  Non-Drowsy: {non_drowsy_count}")
    
    else:
        print(f"✗ Path not found: {input_path}")


if __name__ == "__main__":
    main()
