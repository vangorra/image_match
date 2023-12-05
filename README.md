# image_match

A tool for finding images inside of other images.
The purpose of this tool was to use a camera to determine if a door on a henhouse
is open or not (hence the test images). When it comes to home automation, this method tends to be more 
reliable than using a battery powered sensor.

## Quick Start

### CLI
```sh
# One-off CLI call
$ docker run \
    --rm \
    ghcr.io/vangorra/image_match:main \
    match \
    --reference-dir "./reference/chicken_door" \
    --sample-url "rtsp://10.30.11.17:8554/henhouse_record" \
    --crop-x 0.0 \
    --crop-y 0.6 \
    --crop-width 0.05 \
    --crop-height 0.15

# Fetch a cropped reference image
# When placed in a loop over time, is useful for gathering a corpus of reference images.
$ docker run \
    --rm \
    ghcr.io/vangorra/image_match:main \
    fetch \
    --output-file "./my_new_reference_image.png" \
    --sample-url "rtsp://10.30.11.17:8554/henhouse_record" \
    --crop-x 0.0 \
    --crop-y 0.6 \
    --crop-width 0.05 \
    --crop-height 0.15
```

### REST
Create config.yaml
```yaml
---
# debug: False
match_configs:
  chicken_door:
    # Directory containing images that are references of what is to be searched for.
    # Each reference image will be iterated until one or none matches.
    reference_dir: "./reference/chicken_door" # Required
    # Supports static images from URL, static images from filesystem, or RTSP streams (grabs first frame).
    sample_url: "rtsp://10.30.11.17:8554/henhouse_record" # Required
    # Cropping the sample helps speed up matching and increases confidence in matches.
    transform_config:
      # match_mode: flann # (flann or brute_force) default: flann 
      # match_confidence: 0.9 # default: 0.9
      x: 0.0 # default: 0.0
      y: 0.6 # default: 0.0
      width: 0.05 # default: 1.0
      height: 0.15 # default: 1.0
```

```sh
$ docker run \
    --rm \
    --name image_match \
    --volume <path to config.yaml>:/config.yaml:ro \
    --publish 5000:5000 \
    --interactive \
    ghcr.io/vangorra/image_match:main serve --config /config.yaml
$ curl http://localhost:5000/match/chicken_door
```


## Local Build

### Iterative coding
```sh
# Install pipx.
$ sudo apt install pipx

# Build
$ ./scripts/build.sh

# Test
$ ./scripts/test.sh
```

### Build local container

```sh
# Build/test
# The CI argument simluates the build on github. As in, static analysis will check code but not change it.
$ docker build --tag image_match --build-arg CI=1 .

# Run
$ docker run --rm image_match
```
