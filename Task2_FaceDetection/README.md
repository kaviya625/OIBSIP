from google.colab import files
import cv2
import matplotlib.pyplot as plt

# Step 1: Upload Image
uploaded = files.upload()   # This will prompt you to choose an image from your PC

# Step 2: Read the uploaded image
for fn in uploaded.keys():
    img_path = fn  # get uploaded file name
    print("Image uploaded:", img_path)

img = cv2.imread(img_path)

# Step 3: Load Haarcascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Step 4: Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 5: Detect Faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"✅ Found {len(faces)} face(s) in the image")

# Step 6: Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Step 7: Show Image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
