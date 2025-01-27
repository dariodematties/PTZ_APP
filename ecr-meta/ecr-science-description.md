# Science

This application leverages an NVIDIA XAVIER (or comparable edge hardware) to perform real-time object detection and zooming using a PTZ (Pan‐Tilt‐Zoom) camera. By automatically detecting and focusing on objects such as deer, other wildlife, or notable scene elements, the system provides key insights for ecological monitoring, wildlife management, and situational awareness. Its scientific relevance lies in continuous, unobtrusive observation of dynamic environments—helping researchers and practitioners gather high-quality images for biodiversity studies, behavior research, and timely interventions in remote or hard-to-reach locations.

# AI@Edge

The application deploys the **Florence v2** base model on the edge device. Florence v2 is a powerful vision-language model capable of identifying a variety of objects and scenes. The workflow is:

1. The camera rotates (pan/tilt) and zooms in pre-determined or incremental steps to scan the environment.  
2. Live frames are captured and processed by the Florence v2 model.  
3. If an “interesting” object or scene is detected, the system automatically adjusts the PTZ camera to center and maximize the object in the frame.  
4. A picture is taken and sent to the Sage cloud infrastructure for further processing, archiving, or real-time alerts.

By pushing this AI capability to the edge, the system operates continuously with minimal latency and reduced bandwidth usage—uploading only relevant snapshots rather than a constant video feed.

# Arguments

The application supports the following command-line arguments:

- **`--keepimages` / `-ki`**  
  Keep collected images in a persistent folder for later use. (Boolean flag)

- **`--iterations` / `-it`**  
  Number of iterations (PTZ rounds) to run. (Default: 5)

- **`--object` / `-obj`**  
  The name of the target object of interest to look for (e.g., “animal”). (Default: "animal")

- **`--username` / `-un`**  
  Username for the PTZ camera.

- **`--password` / `-pw`**  
  Password for the PTZ camera.

- **`--cameraip` / `-ip`**  
  IP address of the PTZ camera.

Example usage:
```bash
python main.py -ki -it 5 -obj "deer" -un admin -pw secret -ip xxx.xxx.x.xx
```

# Ontology

The interesting images collected by the system are tagged with metadata for easy retrieval and analysis.
