import cv2
import os

# get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

img_dir = os.path.join(
    BASE_DIR,
    "data",
    "fracture_dataset",
    "fractured"
)

# list files in directory
files = os.listdir(img_dir)
print("Files found:", files[:5])  # print first few files only

# pick the first image file
img_name = files[0]
img_path = os.path.join(img_dir, img_name)

print("Trying to load:", img_path)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found or could not be read")
else:
    cv2.imshow("Sample X-ray", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Displayed image:", img_name)
