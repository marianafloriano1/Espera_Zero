import cv2
import threading
from flask import Flask, render_template

# Variáveis globais
total_pessoas = 0
tempo_total = 0   # <-- aqui vamos armazenar os minutos calculados
next_id = 0
tracked_people = []

MAX_LOST_FRAMES = 5
DETECT_INTERVAL = 30

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def contar_pessoas():
    global total_pessoas, tempo_total, next_id, tracked_people
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % DETECT_INTERVAL == 0 or len(tracked_people) == 0:
            boxes, _ = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
            for (x, y, w, h) in boxes:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracked_people.append({'id': next_id, 'tracker': tracker, 'lost_frames': 0})
                next_id += 1

        new_tracked = []
        for person in tracked_people:
            success, box = person['tracker'].update(frame)
            if success:
                person['lost_frames'] = 0
                new_tracked.append(person)
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                person['lost_frames'] += 1
                if person['lost_frames'] < MAX_LOST_FRAMES:
                    new_tracked.append(person)
        tracked_people = new_tracked

        # quantidade de pessoas atuais
        total_pessoas = len(tracked_people)

        # cálculo automático -> cada pessoa = 5 minutos
        tempo_total = total_pessoas * 5  

        # exibe no frame
        cv2.putText(frame, f"Pessoas: {total_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Tempo: {tempo_total} min", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Detecção de Pessoas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/opcoes")
def opcoes():
    return render_template("opcoes.html")

@app.route("/preferencial")
def preferencial():
    return render_template("preferencial.html")

@app.route("/rapido")
def rapido():
    return render_template("rapido.html")

@app.route("/autoatendimento")
def autoatendimento():
    return render_template("autoatendimento.html")

@app.route("/normal")
def normal():
    return render_template("normal.html")

@app.route("/tv")
def tv():
    return render_template("tv.html")

@app.route("/count")
def count():
    return f"Pessoas: {total_pessoas} | Tempo total: {tempo_total} minutos"


if __name__ == "__main__":
    threading.Thread(target=contar_pessoas, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

