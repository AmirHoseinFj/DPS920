![Demo](recording.gif)

# [Click for Full Recording](Recording.mp4)

# Self-Driving Car Simulation Project (CNN) - Amirhossein Ghaffarzadeh 120734223 - Group 16 

Trains an Nvidia end-to-end CNN to predict steering angle from the Udacity
self-driving car simulator's center-camera images, then drives the car
autonomously in the simulator using the trained model.


## Project structure

```
finalproject/
├── driving_log.csv        # recorded driving data (Center,Left,Right,Steering,Throttle,Brake,Speed)
├── IMG/                    # recorded camera frames
├── TestSimulation.py       # inference script: connects to the simulator, drives using model.h5
├── model.h5                 # trained model (produced by src/train.py)
├── package_list.txt        # conda environment spec
└── src/
    ├── preprocessing.py     # crop / YUV / blur / resize / normalize (shared by train + inference)
    ├── augmentation.py      # pan / zoom / rotate / brightness / flip (train only)
    ├── data_utils.py        # CSV loading, steering-histogram balancing, batch generator
    ├── model.py              # Nvidia CNN architecture
    └── train.py               # training entry point
```

## 1. Environment setup

The working environment is a **Python 3.10 venv**:

```
py -3.10 -m venv env310
env310\Scripts\python.exe -m pip install -r requirements.txt
```

Every command below uses `env310\Scripts\python.exe` explicitly, so no
`activate` step is needed.

Use this same environment for both training and running `TestSimulation.py`,
mixing library versions between the two is a common source of simulator
failures. Been there, done that.

### Why not `package_list.txt`?

`package_list.txt` is the original conda spec, pinned for **Python 3.8**. Its
pypi pins cannot be installed into a Python 3.10 interpreter, installing them
produces an environment that imports cleanly for TensorFlow/NumPy/OpenCV but
dies the moment `TestSimulation.py` starts:

```
eventlet 0.25.1   -> TypeError: cannot set 'is_timeout' attribute of
                     immutable type 'TimeoutError'
python-socketio   -> fails transitively through the same eventlet import
```

Python 3.10 made `TimeoutError` immutable, and eventlet only stopped patching
it in 0.33. `requirements.txt` pins the oldest versions that both run on
Python 3.10 and still speak the protocol the simulator expects.

### The Socket.IO version trap

This is the single most time-consuming part of the setup, so it is worth talking about it: **the simulator only works with `python-socketio` 4.x.**

The Unity client is built on `websocket-sharp` (visible as
`User-Agent: websocket-sharp/1.0` in the request headers). It opens a websocket
directly and does not complete the namespace handshake that `python-socketio`
5.x expects. Against a 5.x server the transport upgrade succeeds, telemetry
arrives, and then every event is discarded with:

```
None is not connected to namespace /
```

The only packet the server ever sends back is the `throttle: 0` from
`connect()`, so **the car sits still and nothing looks obviously broken.**
That symptom is identical to the one you get from a version that is too old,
which is what makes this confusing to diagnose. The pins that work:

| package | version | why |
|---|---|---|
| `python-socketio` | 4.6.1 | namespace behaviour the simulator expects |
| `python-engineio` | 3.13.2 | matching Engine.IO v3 |
| `eventlet` | 0.33.3 | oldest series that runs on Python 3.10 |

Do **not** pair `python-socketio` 4.6.1 with `python-engineio` 4.x to split the
difference, pip blocks it, and forcing it with `--no-deps` gives a server that
returns HTTP 401 on every handshake.

Working output looks like this:

```
throttle=+1.000  steering=-0.053  speed=0.00
throttle=+0.451  steering=-0.061  speed=5.02
```

### GPU note

TensorFlow **dropped native Windows GPU support after 2.10**, which is one
reason this project pins 2.10.1, it uses the GPU directly, with no WSL2 needed
(verified on an RTX 3070: `tf.config.list_physical_devices('GPU')` is non-empty).

The speed-up is modest, because the problem is not the GPU. Profiling showed
~216 ms/step against ~49 ms/step of pure compute, so roughly **77% of each step
is the batch generator** doing JPEG decode and OpenCV augmentation on one Python
thread. A 15-epoch run takes about 16 minutes either way. If you want training to
be faster, parallelising the generator would help far more than the GPU
does.

### Keras 2 vs Keras 3

TF 2.10 ships Keras 2, and `model.h5` is saved in that format. A model saved by
Keras 3 will **not** load here, it fails with
`Unrecognized keyword arguments: ['batch_shape']`.

`TestSimulation.py` loads with `load_model('model.h5', compile=False)`, which
skips restoring the optimizer state. That is harmless because only `predict()`
is used at inference time.


## 2. Data collection

1. Launched the simulator (`beta_simulator_windows`), chose Training Mode.
2. Drove the leftmost track, ~5 laps forward + ~5 laps in reverse, using the
   **Keyboard**, I couldn't do it with mouse, was it even available?
3. Turned on Recording and selected an output folder before driving.
4. This produced `IMG/` (camera frames) and `driving_log.csv` (steering log).


## 3. Training

From the project root:

```
env310\Scripts\python.exe src/train.py
```

Useful flags (all optional, shown with defaults):

```
python src/train.py --csv driving_log.csv --img-dir IMG \
    --batch-size 100 --steps-per-epoch 300 --epochs 15 \
    --max-per-bin 400 --correction 0.2 --out model.h5
```

### Why all three cameras are used

The simulator records a **left, center and right** image every frame, and the
first version of this project trained on the center image only, which, after
histogram balancing, left just 880 training images out of 3,797 recorded frames.
The car drove but wandered out of the lane.

The fix is the recovery trick from the Nvidia paper I pulled. The side cameras see the
road from a position offset from the car's center, so labelling a left image
with `steering + 0.2` and a right image with `steering - 0.2` teaches the model
to steer back to the middle whenever it drifts. It needs no extra driving, those images were already on disk and it triples the dataset:

| | center only | all three cameras |
|---|---|---|
| samples before balancing | 3,797 | 11,391 |
| training images after balancing | 880 | 2,816 |

Pass `--center-only` to disable this and reproduce the old behaviour.

What it does:

1. Loads `driving_log.csv`.
2. Folds the left/right cameras in as extra samples with a `--correction`
   steering offset (skip with `--center-only`).
3. Balances the steering-angle distribution by capping samples per histogram
   bin (`--max-per-bin`), so the model isn't dominated by near-zero steering
   angles from straight driving. Saves a before/after histogram to
   `steering_histogram.png`.
5. Splits into train/validation sets.
6. Trains via a batch generator that applies random augmentation (pan, zoom,
   rotate, brightness, horizontal flip) to the **training** split only, then
   applies the shared preprocessing (crop → YUV → Gaussian blur → resize to
   200×66 → normalize) to every image.
7. Saves the **best** epoch by validation loss to `model.h5` (via
   `ModelCheckpoint`, not the last epoch, validation loss bounces between
   epochs, so the final one is often not the best), and a loss curve to
   `training_history.png`.

### Augmentation should correct the steering label

Any augmentation that moves the road sideways changes what the correct steering
angle is. An earlier version applied `pan` and `rotate` without touching the
label, which taught the network that a road sitting off to one side still means
"drive straight", visible in the simulator as slow drift out of the lane.

`src/augmentation.py` now returns a corrected angle from both:

- `pan` - shifts by a fraction of the image width, adjusts the label by
  `shift_fraction * 0.4`
- `rotate` - adjusts by `angle_degrees * 0.006`
- `flip` - negates the angle (was already correct)
- `zoom` / `brightness` - centred or colour-only, so the label is left alone

## 4. Testing in the simulator

1. Use the same `env310` environment used for training.
2. Run, from the project root:
   ```
   env310\Scripts\python.exe TestSimulation.py
   ```
   Wait for `wsgi starting up on http://0.0.0.0:4567` before continuing.
3. Launch the simulator and switch to **Autonomous Mode** with the same
   track/settings used for data collection.
4. The script preprocesses each incoming frame with the exact same
   `src/preprocessing.py` used during training, predicts a steering angle,
   and sends `(steering, throttle)` back to the simulator over Socket.IO.

## Notes

- `TestSimulation.py` imports `preprocess_image` from `src/preprocessing.py`
  rather than duplicating the logic, so training and inference preprocessing
  can never silently drift apart.
- `maxSpeed` in `TestSimulation.py` controls the throttle curve
  (`throttle = 1 - speed / maxSpeed`); lower it for a more cautious drive.
