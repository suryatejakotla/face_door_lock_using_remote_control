import face_recognition
import pickle
import cv2
import imutils

# Path to encodings file
encodingsP = "encodings.pickle"
image_path = "dataset/1.jpeg"  # Change this to the image path

# Load known face encodings
print("[INFO] Loading encodings...")
data = pickle.loads(open(encodingsP, "rb").read())

# Load the input image
image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Image not found at: {image_path}")
    exit()

# Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect face locations in the image
boxes = face_recognition.face_locations(rgb_image, model="hog")  # Change to "cnn" for better accuracy

# Compute face encodings for the detected faces
encodings = face_recognition.face_encodings(rgb_image, boxes)

# List to store recognized names
names = []

# Loop over the facial encodings
for encoding in encodings:
    # Compare faces and find best match
    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
    name = "Unknown"  # Default name if no match is found

    # Check if we found a match
    if True in matches:
        # Get indexes of matched faces and count occurrences
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # Get the most recognized name
        name = max(counts, key=counts.get)

    names.append(name)

# Draw rectangles and labels on the image
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Display the result
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
