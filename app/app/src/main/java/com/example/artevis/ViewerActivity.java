package com.example.artevis;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ViewerActivity extends AppCompatActivity implements SensorEventListener {

    private String project = "bauhaus";
    private ImageView imageView;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    // App state mapping to asset filename prefix: inverted-reverted-w1-w2-w3
    static class State {
        boolean inverted = false;
        boolean reverted = false;
        int w1 = 0, w2 = 0, w3 = 0; // 0..3

        String prefix() {
            return String.format(Locale.US, "%d-%d-%d-%d-%d",
                    inverted ? 1 : 0, reverted ? 1 : 0, w1, w2, w3);
        }
    }

    private final State state = new State();

    // Sensors
    private SensorManager sensorManager;
    private Sensor rotationVector;
    private Sensor linearAcceleration;
    private Sensor accelerometer;

    // Rotation/yaw baseline to control w3
    private float baseYawDeg = Float.NaN;

    // Tilt latch for invert toggle
    private boolean tiltLatched = false;

    // Movement cooldown for odometer steps
    private long lastMoveMs = 0L;
    private static final long MOVE_COOLDOWN_MS = 250L;

    // Shake detection
    private long lastShakeMs = 0L;
    private static final long SHAKE_COOLDOWN_MS = 1200L;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_viewer);

        project = getIntent().getStringExtra("project");
        if (project == null) project = "bauhaus";

        imageView = findViewById(R.id.fullImage);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        updateImage(); // initial
    }

    @Override
    protected void onResume() {
        super.onResume();
        int rate = SensorManager.SENSOR_DELAY_GAME;
        if (rotationVector != null) sensorManager.registerListener(this, rotationVector, rate);
        if (linearAcceleration != null) sensorManager.registerListener(this, linearAcceleration, rate);
        if (accelerometer != null) sensorManager.registerListener(this, accelerometer, rate);
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    @Override
    public void onBackPressed() {
        // Return to image selection
        super.onBackPressed();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
            handleRotationVector(event);
        } else if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            handleLinearAcceleration(event);
        } else if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            handleShake(event);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    private void handleRotationVector(SensorEvent event) {
        float[] rotationMatrix = new float[9];
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values);
        float[] orientation = new float[3];
        SensorManager.getOrientation(rotationMatrix, orientation);
        // Azimuth (yaw), Pitch, Roll in radians
        float azimuth = orientation[0];
        float pitch = orientation[1];
        float roll = orientation[2];

        float yawDeg = (float)Math.toDegrees(azimuth);
        float pitchDeg = (float)Math.toDegrees(pitch);
        float rollDeg = (float)Math.toDegrees(roll);

        if (Float.isNaN(baseYawDeg)) {
            baseYawDeg = yawDeg;
        }

        // Map yaw delta to w3 (last weight), 4 sectors (each 90 degrees)
        float deltaYaw = normalizeAngleDeg(yawDeg - baseYawDeg);
        int sector = (int) Math.round(deltaYaw / 90f);
        sector = ((sector % 4) + 4) % 4;
        if (sector != state.w3) {
            state.w3 = sector;
            updateImage();
        }

        // Tilt invert toggle: when pitch or roll exceeds 30°, toggle once; unlock when below 10°
        float tiltMag = Math.max(Math.abs(pitchDeg), Math.abs(rollDeg));
        if (!tiltLatched && tiltMag > 30f) {
            state.inverted = !state.inverted;
            tiltLatched = true;
            updateImage();
        } else if (tiltLatched && tiltMag < 10f) {
            tiltLatched = false;
        }
    }

    private void handleLinearAcceleration(SensorEvent event) {
        float ax = event.values[0];
        float ay = event.values[1];
        float az = event.values[2]; // device Z axis: forward/back

        long now = SystemClock.elapsedRealtime();
        if (now - lastMoveMs < MOVE_COOLDOWN_MS) return;

        // Forward/backward impulse threshold
        final float THRESH = 2.5f;
        if (az > THRESH) {
            // forward
            lastMoveMs = now;
            incrementOdometer(+1);
        } else if (az < -THRESH) {
            // backward
            lastMoveMs = now;
            incrementOdometer(-1);
        }
    }

    private void handleShake(SensorEvent event) {
        // Use raw accelerometer to detect strong shakes (including gravity)
        float ax = event.values[0];
        float ay = event.values[1];
        float az = event.values[2];
        double magnitude = Math.sqrt(ax*ax + ay*ay + az*az);
        long now = SystemClock.elapsedRealtime();
        if (magnitude > 18.0 && now - lastShakeMs > SHAKE_COOLDOWN_MS) {
            lastShakeMs = now;
            randomize();
        }
    }

    private void incrementOdometer(int dir) {
        // dir +1 forward, -1 backward
        int oldW1 = state.w1;
        int oldW2 = state.w2;
        if (dir > 0) {
            state.w1 = (state.w1 + 1) % 4;
            if (state.w1 == 0) {
                state.w2 = (state.w2 + 1) % 4;
                if (state.w2 == 0) {
                    state.reverted = !state.reverted; // swap 1<->3
                }
            }
        } else {
            state.w1 = (state.w1 + 3) % 4; // -1 mod 4
            if (oldW1 == 0) { // wrapped from 0 to 3
                state.w2 = (state.w2 + 3) % 4;
                if (oldW2 == 0) { // wrapped from 0 to 3
                    state.reverted = !state.reverted;
                }
            }
        }
        updateImage();
    }

    private void randomize() {
        Random rnd = new Random();
        state.inverted = rnd.nextBoolean();
        state.reverted = rnd.nextBoolean();
        state.w1 = rnd.nextInt(4);
        state.w2 = rnd.nextInt(4);
        state.w3 = rnd.nextInt(4);
        // Reset yaw baseline so w3 remains consistent with physical rotation:
        baseYawDeg = Float.NaN;
        updateImage();
    }

    private static float normalizeAngleDeg(float a) {
        while (a <= -180f) a += 360f;
        while (a > 180f) a -= 360f;
        return a;
    }

    private void updateImage() {
        final String assetPath = "images/" + project + "/" + state.prefix() + "_art.jpg";
        loadAssetBitmapAsync(getAssets(), assetPath, imageView);
    }

    private void loadAssetBitmapAsync(final AssetManager am, final String assetPath, final ImageView target) {
        final int reqW = target.getWidth() > 0 ? target.getWidth() : 1440;
        final int reqH = target.getHeight() > 0 ? target.getHeight() : 3040;
        // Ensure portrait-friendly orientation by rotating landscape images 90°
        // so they fill more of the portrait screen.
        executor.submit(new Runnable() {
            @Override public void run() {
                final Bitmap raw = decodeSampledBitmapFromAsset(am, assetPath, reqW, reqH);
                final Bitmap bmp = rotateIfLandscape(raw);
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        if (!isFinishing()) {
                            target.setImageBitmap(bmp);
                        }
                    }
                });
            }
        });
    }

    public static Bitmap decodeSampledBitmapFromAsset(AssetManager am, String assetPath, int reqWidth, int reqHeight) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try (InputStream is = am.open(assetPath)) {
            BitmapFactory.decodeStream(is, null, options);
        } catch (IOException e) {
            return null;
        }
        options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
        options.inJustDecodeBounds = false;
        try (InputStream is = am.open(assetPath)) {
            return BitmapFactory.decodeStream(is, null, options);
        } catch (IOException e) {
            return null;
        }
    }

    public static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        int height = options.outHeight;
        int width = options.outWidth;
        int inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    private Bitmap rotateIfLandscape(Bitmap bmp) {
        if (bmp == null) return null;
        if (bmp.getWidth() > bmp.getHeight()) {
            android.graphics.Matrix m = new android.graphics.Matrix();
            m.postRotate(90);
            try {
                Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
                if (rotated != bmp) {
                    bmp.recycle();
                }
                return rotated;
            } catch (OutOfMemoryError e) {
                return bmp; // fallback without rotation
            }
        }
        return bmp; // portrait or square
    }
}