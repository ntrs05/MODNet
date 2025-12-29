# Images

First of all, you need to install the dependencies by using the command:
```
pip install -r requirements.txt
```

To run the script, paste this line into Terminal when you are inside the `MODNet`directory:
```
py -m demo.image_matting.colab.inference --input-path "input/<your_image_file>" --output-path "result" --ckpt-path "pretrained/modnet_photographic_portrait_matting.ckpt"
```

For example, to delete background for `alonso.jpg`, put it into the `input` folder, then use this command to run: 
```
py -m demo.image_matting.colab.inference --input-path "input/alonso.jpg" --output-path "result" --ckpt-path "pretrained/modnet_photographic_portrait_matting.ckpt"
```

A new file will appear in the `result` folder, it's a new image without the background named `alonso_foreground.png`

You can do this with multiple images, just put how many images you want into the `input` the folder and run the command but the `--input-path` now is `input/` only:
```
py -m demo.image_matting.colab.inference --input-path "input/" --output-path "result" --ckpt-path "pretrained/modnet_photographic_portrait_matting.ckpt"
```
