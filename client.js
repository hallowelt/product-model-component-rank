import io from 'socket.io-client';

const tfvis = require('@tensorflow/tfjs-vis');
const tf = require('@tensorflow/tfjs');
const somethingButton = document.getElementById('something-button');

const socket =
    io('http://localhost:8002',
       {reconnectionDelay: 300, reconnectionDelayMax: 300});

var lossTracker = [];
var mseTracker = [];
var index = 0;

function resetValues() {
  lossTracker = [];
  mseTracker = [];
  index = 0;
}

somethingButton.onclick = () => {
  somethingButton.disabled = true;
  resetValues();
  socket.emit( 'startTraining' );
};

// functions to handle socket events
socket.on('connect', () => {
	console.log( "connected" );
    document.getElementById('waiting-msg').style.display = 'none';
});

socket.on('disconnect', () => {
	console.log( "disconnected" );
    document.getElementById('waiting-msg').style.display = 'block';
});

// functions to handle socket events
socket.on('startingTraining', () => {
	console.log( "Started training" );
    document.getElementById('showResult').innerHTML = "Started training";
});

socket.on('statusUpdate', (message) => {
	console.log( "statusUpdate: " + message );
  document.getElementById('showResult').innerHTML = message;
});

socket.on('modelReady', async (model) => {
  const modelObj = await unserializeModel( model );

  const containerSummary = document.getElementById( 'showSummary' );
  tfvis.show.modelSummary( containerSummary, modelObj );
  const layer1Summary = document.getElementById( 'layer1Summary' );
  tfvis.show.layer(layer1Summary, modelObj.getLayer(undefined, 0));
  const layer2Summary = document.getElementById( 'layer2Summary' );
  tfvis.show.layer(layer2Summary, modelObj.getLayer(undefined, 1));
});

async function unserializeModel( model ) {
  const json = JSON.parse(model);
  const weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
  const modelObj = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));

  return modelObj;
}

socket.on('doneSomething', (result) => {
	document.getElementById('showResult').innerHTML = "Finished";
  somethingButton.disabled = false;
  console.log( result );
  const container = document.getElementById( 'showDiagram' );
	tfvis.render.scatterplot(
	  container,
	  {values: [result.o, result.p], series: ["test", "predicted"]},
	  {
		  xLabel: 'Horsepower',
		  yLabel: 'MPG',
		  height: 300
	  }
	);
});

socket.on( 'epochEnd', ( epoch, logs) => {
  console.log( logs.loss );
  console.log( logs.mse );
  lossTracker.push( {x: index, y: logs.loss} );
  mseTracker.push( {x: index, y: logs.mse} );
  index++;
  const container = document.getElementById( 'showProgress' );
  tfvis.render.linechart(
	  container,
	  {values: [lossTracker, mseTracker], series: ["loss", "mse"]}
	);
});