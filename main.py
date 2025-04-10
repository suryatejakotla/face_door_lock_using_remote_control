

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import serial
import imaplib
import email
from email.header import decode_header
import smtplib
import threading

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "person not found"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize video stream and allow the camera sensor to warm up
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)

# Initialize serial connection to Arduino
arduino = serial.Serial('COM7', 9600, timeout=1)  # Change 'COM3' to your Arduino's port
time.sleep(2)  # Allow time for the connection to establish

# Email configuration
EMAIL = "purplevja@gmail.com"  # Replace with your email
PASSWORD = "pahy quyr cbxq cvmu"        # Replace with your email password
SMTP_SERVER = "smtp.gmail.com"
IMAP_SERVER = "imap.gmail.com"
SMTP_PORT = 587
IMAP_PORT = 993

# Function to check emails and process commands
def check_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")

        status, messages = mail.search(None, '(UNSEEN)')
        if status == "OK":
            for num in messages[0].split():
                status, msg = mail.fetch(num, '(RFC822)')
                if status == "OK":
                    for response in msg:
                        if isinstance(response, tuple):
                            msg = email.message_from_bytes(response[1])
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8")
                            if subject.lower() == "open":
                                arduino.write(b'B')  # Send 'B' to Arduino
                                print("Email Door Open")
                            elif subject.lower() == "close":
                                arduino.write(b'C')  # Send 'C' to Arduino
                                print("Email Door Close")
        mail.logout()
    except Exception as e:
        print(f"Error checking email: {e}")

# Function to run email checking in a separate thread
def check_email_periodically():
    while True:
        check_email()
        time.sleep(10)  # Check emails every 10 seconds

# Start the email-checking thread
email_thread = threading.Thread(target=check_email_periodically, daemon=True)
email_thread.start()

# Start the FPS counter
fps = FPS().start()

# Loop over frames from the video file stream
while True:
    # Grab the frame from the threaded video stream and resize it to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # Detect the face boxes
    boxes = face_recognition.face_locations(frame)
    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "person not found"  # If face is not recognized, print Unknown

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces and initialize a dictionary to count
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number of votes
            name = max(counts, key=counts.get)

            # If someone in the dataset is identified, print their name
            if currentname != name:
                currentname = name
                print("Recognized:", currentname)
                arduino.write(b'A')  # Send 'A' to Arduino
                print("Door Open")

        # Update the list of names
        names.append(name)

    # Loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when 'q' key is pressed
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
arduino.close()