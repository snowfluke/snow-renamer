require("@tensorflow/tfjs-backend-cpu");
require("@tensorflow/tfjs-backend-webgl");

const fs = require("fs");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const tf = require("@tensorflow/tfjs-node");

const directoryPath = process.argv[2];

async function main() {
  if (!directoryPath)
    return console.error(
      "Please specify a directory path as the first argument."
    );

  console.log("Loading the models ... (this may takes few minutes)");
  const model = await cocoSsd.load();

  console.log("Model loaded successfully ...");
  const files = fs.readdirSync(directoryPath);

  for (const file of files) {
    const extension = file.split(".").pop().toLowerCase();
    if (extension !== "jpg" && extension !== "png") continue;

    const buffer = fs.readFileSync(`${directoryPath}/${file}`);
    const tensor = tf.node.decodeImage(buffer);

    const predictions = await model.detect(tensor);
    const objects = predictions.map((p) => p.class);

    const caption = objects.join(" ").trim();
    let newName = caption.replace(/[^a-z0-9]/gi, "_");

    if (!newName) newName = "unknown";
    newName = newName + `_${randomNumber()}.${extension}`;

    fs.renameSync(`${directoryPath}/${file}`, `${directoryPath}/${newName}`); // rename file
    console.log(`----> Renamed "${file}" to "${newName}"`);
  }
}

function randomNumber() {
  return Math.floor(Math.random() * 9999) + 1;
}

main().catch(console.error);
