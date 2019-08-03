const assetURL = chrome.runtime.getURL("/asset");
const modelURL = chrome.runtime.getURL("/models");
const loadModel = Promise.all([faceapi.nets.faceRecognitionNet.loadFromUri(modelURL), faceapi.nets.faceLandmark68Net.loadFromUri(modelURL), faceapi.nets.ssdMobilenetv1.loadFromUri(modelURL)]);

const imagesGlobal = new Set();
const positionMemory = Object.create(null);
const minResolution = 80;

chrome.runtime.sendMessage("RequestLabeledFaceDescriptors", start);

async function start({ labeledFaceDescriptors, constants: { threshold } }) {
  await loadModel;
  labeledFaceDescriptors = labeledFaceDescriptors.map(({ label, descriptions }) => new faceapi.LabeledFaceDescriptors(label, descriptions.map(description => Float32Array.from(description))));
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, threshold);

  setInterval(async () => {
    const $images = $("img");

    for (const image of $images) {
      const $image = $(image);
      if (!imagesGlobal.has(image)) {
        const uuid = uuidv4();
        $image.attr("uuid", uuid);
        const { width, height, top, left } = {
          width: $image.width(),
          height: $image.height(),
          ...$image.position(),
        };

        if (width > minResolution && height > minResolution) {
          try {
            const canvas = faceapi.createCanvasFromMedia(image);
            const $canvas = $(canvas)
              .css({
                position: "absolute",
                top,
                left,
              })
              .attr("uuid", uuid)
              .appendTo($image.parent());

            imagesGlobal.add(image);
            positionMemory[uuid] = `${width}-${height}-${top}-${left}`;

            const displaySize = {
              width,
              height,
            };

            const src = $image.attr("src");
            const _image = await faceapi.fetchImage(src);

            faceapi.matchDimensions(canvas, displaySize);
            const detections = await faceapi
              .detectAllFaces(_image)
              .withFaceLandmarks()
              .withFaceDescriptors();

            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
            results.forEach((result, i) => {
              const box = resizedDetections[i].detection.box;
              const label = result.toString();

              if (result.distance < threshold) {
                // const drawBox = new faceapi.draw.DrawBox(box, { label });
                // drawBox.draw(canvas);
                const { _height, _width, _x, _y } = box;
                const ctx = canvas.getContext("2d");
                ctx.fillStyle = "black";
                ctx.fillRect(_x, _y, _width, _height);
              }
            });
          } catch (_) {
            console.log("Error:", _);
          }
        }
      } else {
        const uuid = $image.attr("uuid");
        const { width, height, top, left } = {
          width: $image.width(),
          height: $image.height(),
          ...$image.position(),
        };
        if (positionMemory[uuid] !== `${width}-${height}-${top}-${left}`) {
          const $overlay = $(`canvas[uuid="${uuid}"]`).css({
            position: "absolute",
            width,
            height,
            top,
            left,
          });
          positionMemory[uuid] = `${width}-${height}-${top}-${left}`;
        }
      }
    }
  }, 1000);
}

// https://stackoverflow.com/questions/105034/create-guid-uuid-in-javascript
function uuidv4() {
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c => (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16));
}
