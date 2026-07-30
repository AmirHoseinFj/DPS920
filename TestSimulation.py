import os
import sys
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

from preprocessing import preprocess_image

# This must run on python-socketio 4.x / python-engineio 3.x. The simulator is
# a Unity build using websocket-sharp, and it does not complete the namespace
# handshake that python-socketio 5.x requires -- against a 5.x server every
# telemetry event is dropped with "None is not connected to namespace /" and
# the car just sits there. See README for the exact pins.
#
# Set logger/engineio_logger to True to see the raw packet exchange when
# debugging a connection problem; they are noisy enough to bury the
# steering/throttle telemetry printed below.
sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)

maxSpeed = 10
minSpeed = 4
# How hard to back off the throttle mid-corner. The model steers far better
# when it is not simultaneously accelerating into a turn, so scale the target
# speed down as the predicted steering angle grows.
corneringSlowdown = 0.6


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess_image(image)
    image = np.array([image])
    # predict() returns shape (1, 1); index the single element rather than
    # calling float() on the array -- NumPy deprecated that coercion in 1.25
    # and it raises in newer versions.
    steering = float(model.predict(image, verbose=0)[0][0])
    steering = float(np.clip(steering, -1.0, 1.0))

    # Ease off in corners: a tight predicted angle lowers the target speed,
    # which keeps the car from understeering wide on the sharp bends.
    targetSpeed = maxSpeed * (1.0 - corneringSlowdown * min(abs(steering), 1.0))
    targetSpeed = max(targetSpeed, minSpeed)
    throttle = float(np.clip(1.0 - speed / targetSpeed, -1.0, 1.0))

    print(f'throttle={throttle:+.3f}  steering={steering:+.3f}  speed={speed:.2f}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected', sid)
    sendControl(0, 0)
    return True


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle' : steering.__str__(),
        'throttle' : throttle.__str__()
    })

if __name__ == "__main__":
    model = load_model('model.h5', compile=False)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)