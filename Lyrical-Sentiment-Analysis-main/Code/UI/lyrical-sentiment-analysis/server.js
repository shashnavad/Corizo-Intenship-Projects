const { spawn } = require('child_process')

const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'build')));

app.post("/classify", async (req, res) => {
	try {
		if (!(
			"model" in req.body &&
			"lyrics" in req.body
		)) {
			console.error("Invalid parameters");
			res.status(400).json({error: "Invalid parameters"});
			return;
		}

		const pythonProcess = spawn('python', ['./src/models/run_model.py']);
		pythonProcess.stdin.write(JSON.stringify({model: req.body.model, lyrics: req.body.lyrics}));
		pythonProcess.stdin.end();
	
		let data = '';
		pythonProcess.stdout.on('data', (chunk) => {
			data += chunk;
		});
	
		pythonProcess.on('close', (code) => {
			if (code !== 0) {
				return res.status(500).json({ error: 'Internal server error' });
			}
			res.json(JSON.parse(data));
		});
	
		pythonProcess.stderr.on('data', (data) => {
			console.error(`stderr: ${data}`);
		});


	} catch (error) {
		console.error('Error executing query', error);
		res.status(500).json({ error: 'Internal server error' });
	}
});

app.listen(port, () => {
	console.log(`Server is running on port ${port}`);
});