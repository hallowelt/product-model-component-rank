const tf = require('@tensorflow/tfjs-node');
const tfvis = require('@tensorflow/tfjs-vis');
const fetch = require('node-fetch');
const fs = require('fs');
const TRAIN_DATA_PATH = '/home/mglaser/workspace/ml/vis/carsData.json';

var io = null;

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
 async function getData() {
	const carsDataResponse = await fs.readFileSync(TRAIN_DATA_PATH, 'utf8');
	const carsData = await JSON.parse( carsDataResponse );
	const cleaned = carsData.map(car => ({
	  mpg: car.Miles_per_Gallon,
	  horsepower: car.Horsepower,
	}))
	.filter(car => (car.mpg != null && car.horsepower != null));
  
	return cleaned;
  }

  async function run(sample, socket) {

	io = socket;
	// Load and plot the original input data that we are going to train on.
	const data = await getData();
	const values = data.map(d => ({
	  x: d.horsepower,
	  y: d.mpg
	}));

	io.emit( 'statusUpdate', 'Creating model...' );
	const model = await createModel(io);

	// Convert the data to a form we can use for training.
	const tensorData = convertToTensor(data);
	const {inputs, labels} = tensorData;

	// Train the model
	await trainModel(model, inputs, labels);
	console.log('Done Training');

	const jsonStr = await serializeModel( model );
	io.emit( 'modelReady', jsonStr );

	const learningData = testModel(model, data, tensorData);
	console.log('Done Testing');
  
	const jsonStr1 = await serializeModel( model );
	io.emit( 'modelReady', jsonStr1 );

	return {
		v: values,
		m: model,
		o: learningData.o,
		p: learningData.p
	}
	// More code will be added below
  }
  
  async function createModel() {
	// Create a sequential model
	const model = tf.sequential();
  
	// Add a single input layer
//	model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
	model.add(tf.layers.dense({inputShape: [1], units: 10, useBias: true}));

	model.add(tf.layers.dense({units: 1000, activation: 'sigmoid'}));

	model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}));
  
	// Add an output layer
	model.add(tf.layers.dense({units: 1, useBias: true}));

	const jsonStr = await serializeModel( model );

	io.emit( 'modelReady', jsonStr );
	return model;
  }

  async function serializeModel( model ) {
	let result = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
	result.weightData = Buffer.from(result.weightData).toString("base64");
	const jsonStr = JSON.stringify(result);
	return jsonStr;
  }

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
 function convertToTensor(data) {
	// Wrapping these calculations in a tidy will dispose any
	// intermediate tensors.
  
	return tf.tidy(() => {
	  // Step 1. Shuffle the data
	  tf.util.shuffle(data);
  
	  // Step 2. Convert data to Tensor
	  const inputs = data.map(d => d.horsepower)
	  const labels = data.map(d => d.mpg);
  
	  const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
	  const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
	  //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
	  const inputMax = inputTensor.max();
	  const inputMin = inputTensor.min();
	  const labelMax = labelTensor.max();
	  const labelMin = labelTensor.min();
  
	  const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
	  const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
	  return {
		inputs: normalizedInputs,
		labels: normalizedLabels,
		// Return the min/max bounds so we can use them later.
		inputMax,
		inputMin,
		labelMax,
		labelMin,
	  }
	});
  }
  
  async function trainModel(model, inputs, labels) {
	// Prepare the model for training.
	model.compile({
	  optimizer: tf.train.adam(),
	  loss: tf.losses.meanSquaredError,
	  metrics: ['mse'],
	});
  
	const batchSize = 32;
	const epochs = 50;
  
	return await model.fit(inputs, labels, {
	  batchSize,
	  epochs,
	  shuffle: true,
	  callbacks: {
		  'onEpochEnd' : function( epoch, logs ) {
			io.emit( 'epochEnd', epoch, logs );
		  }
	  }
	});
  }

  function testModel(model, inputData, normalizationData) {
	const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
	// Generate predictions for a uniform range of numbers between 0 and 1;
	// We un-normalize the data by doing the inverse of the min-max scaling
	// that we did earlier.
	const [xs, preds] = tf.tidy(() => {
  
	  const xs = tf.linspace(0, 1, 100);
	  const preds = model.predict(xs.reshape([100, 1]));
  
	  const unNormXs = xs
		.mul(inputMax.sub(inputMin))
		.add(inputMin);
  
	  const unNormPreds = preds
		.mul(labelMax.sub(labelMin))
		.add(labelMin);
  
	  // Un-normalize the data
	  return [unNormXs.dataSync(), unNormPreds.dataSync()];
	});
  
  
	const predictedPoints = Array.from(xs).map((val, i) => {
	  return {x: val, y: preds[i]}
	});
  
	const originalPoints = inputData.map(d => ({
	  x: d.horsepower, y: d.mpg,
	}));

	return {
		o: originalPoints,
		p: predictedPoints
	}
  
  }
  
  

module.exports = {
	run
  }