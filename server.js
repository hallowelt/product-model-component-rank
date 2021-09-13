require('@tensorflow/tfjs-node');

const http = require('http');
const socketio = require('socket.io');
const vis = require('./vis');

const PORT = 8002;

// Main function to start server, perform model training, and emit stats via the socket connection
async function run() {
	const port = process.env.PORT || PORT;
	const server = http.createServer();
	const io = socketio(server);
  
	server.listen(port, () => {
	  console.log(`  > Running socket on port: ${port}`);
	});
  
	io.on('connection', (socket) => {
	  socket.on('startTraining', async (sample) => {
		io.emit( 'startedTraining' );
		io.emit( 'doneSomething', await vis.run(sample, io) );
	  });
	});
}

run();