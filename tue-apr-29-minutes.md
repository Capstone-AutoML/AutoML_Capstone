Tuesday, Apr 29

- Assume local storage for data for now (currently using GCS in their org). Potentially cloud access at the end of the project.
- Human intervention UI: open-source ui for relabelling - sth like https://labelstud.io/, port it to the pipeline
- Semi-supervise: have a model inference, if the model “fails”, have human intervention
- -> recommend go wih 2 models: yolo-v8 and a 2nd model. Not fixated on SAM. SAM (maybe a different version from the demo) can do text prompting object detection -> bounding box
- Yolo is both the prelabelling model & the base model to be improved
- - Initial inf -> output will be image that are labelled
- - 20% of raw, with the labels from yolo -> human intervention
- - -> Image folder + label folder for training the base model
- There is already a training script for yolo on the internet, the training part is not important, just make sure it works
- Data augmentation: fine to do it from scratch, or use a library
- Distilled images: just a subset of the images that we already have. We can sample them from the data collection
- Seamless model replacement / inference results continuous feedback: more of an fyi, handled by Vertex AI on GCP, so not of our concern. Will do that with the client together if have time at the end.