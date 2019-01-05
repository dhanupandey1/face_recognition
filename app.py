
from flask import Flask,render_template,request,url_for
import psycopg2
app = Flask(__name__)
# Making connection to the database
conn = psycopg2.connect(database="Facerecognition", user="postgres", password="avnnrd123456", host="localhost",  port="5432")
cur = conn.cursor()
conn.commit()
cur.execute("SELECT name from INFO")
print ("Opened database successfully")
@app.route("/")
def main():
    return render_template('home.html')

# function for  recognising face and sending  at login page

@app.route('/predict', methods=['GET','POST'])
def predict():
    import cv2
    import numpy as np
    import os

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0

    names = cur.fetchall() #fetching names from database
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 70):
                print(id)
                id = names[id]

                confidence = "  {0}%".format(round(100 - confidence))

            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))




            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            if id!='unknown':

                cv2.imwrite("static/frame.jpg", img)

                cam.release()
                cv2.destroyAllWindows()


                return render_template('result.html', prediction=id)
            else:
                cam.release()
                cv2.destroyAllWindows()
                signup()



                return render_template('result1.html')

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
@app.route('/signup', methods=['GET','POST'])
def signup():

    import cv2
    import os
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cur.execute("SELECT id from INFO")
    pid = cur.fetchall()
    face_id = pid[-1][0]
    #face_id = input('\n enter user id end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while (True):
        ret, img = cam.read()
        # img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


    # here we will train our data
    import cv2
    import numpy as np
    from PIL import Image
    import os
    # Path for face image database
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    #return render_template('result.html', prediction=id)

@app.route('/pr', methods=['GET','POST'])
def getdata():
    name = request.form['fname']
    email = request.form['emailid']
    query = "INSERT INTO info(name,email) VALUES(%s,%s)"
    cur.execute(query, (name, email,))
    conn.commit()


if __name__ == "__main__":
    app.run(debug=True)
