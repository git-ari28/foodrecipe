from utils.train import train
from utils.predict import predict
import sys

def main():
    try:
        # Start training
        print("Starting training...")
        train()
        print("Training completed successfully.")

        # Make a prediction after training, if an image path is passed
        if len(sys.argv) > 1:
            # Get image path from command line argument
            image_path = sys.argv[1]
            print(f"Making prediction for image: {image_path}")
            result = predict(image_path)
            print("Generated Recipe:\n", result)
        else:
            print("No image path provided. Skipping prediction.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
