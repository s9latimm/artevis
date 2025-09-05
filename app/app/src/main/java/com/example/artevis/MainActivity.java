package com.example.artevis;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private GridView gridView;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);

    static class ProjectItem {
        final String key;
        final String title;
        final String previewAsset; // images/<key>/0-0-0-0-0_art.jpg
        ProjectItem(String key) {
            this.key = key;
            this.title = key.substring(0,1).toUpperCase() + key.substring(1);
            this.previewAsset = "images/" + key + "/0-0-0-0-0_art.jpg";
        }
    }

    private List<ProjectItem> projects;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Known projects from the web/images folders
        projects = new ArrayList<>(Arrays.asList(
                new ProjectItem("bauhaus"),
                new ProjectItem("fog"),
                new ProjectItem("garden"),
                new ProjectItem("horse"),
                new ProjectItem("jaune"),
                new ProjectItem("mona"),
                new ProjectItem("pearl"),
                new ProjectItem("scream"),
                new ProjectItem("square"),
                new ProjectItem("stars")
        ));

        gridView = findViewById(R.id.gridView);
        gridView.setAdapter(new ProjectAdapter(projects));
        gridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                ProjectItem item = projects.get(position);
                Intent intent = new Intent(MainActivity.this, ViewerActivity.class);
                intent.putExtra("project", item.key);
                startActivity(intent);
            }
        });
    }

    class ProjectAdapter extends BaseAdapter {
        private final List<ProjectItem> items;
        ProjectAdapter(List<ProjectItem> items) { this.items = items; }

        @Override public int getCount() { return items.size(); }
        @Override public Object getItem(int position) { return items.get(position); }
        @Override public long getItemId(int position) { return position; }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ViewHolder holder;
            if (convertView == null) {
                convertView = LayoutInflater.from(MainActivity.this).inflate(R.layout.grid_item, parent, false);
                holder = new ViewHolder();
                holder.imageView = convertView.findViewById(R.id.image);
                holder.titleView = convertView.findViewById(R.id.title);
                convertView.setTag(holder);
            } else {
                holder = (ViewHolder) convertView.getTag();
            }

            ProjectItem item = items.get(position);
            holder.titleView.setText(item.title);
            holder.imageView.setImageBitmap(null);
            loadAssetBitmapAsync(item.previewAsset, holder.imageView, 400, 400);
            return convertView;
        }

        class ViewHolder {
            ImageView imageView;
            TextView titleView;
        }
    }

    private void loadAssetBitmapAsync(final String assetPath, final ImageView imageView, final int reqW, final int reqH) {
        executor.submit(new Runnable() {
            @Override public void run() {
                final Bitmap bmp = decodeSampledBitmapFromAsset(getAssets(), assetPath, reqW, reqH);
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        if (!isFinishing()) {
                            imageView.setImageBitmap(bmp);
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
}