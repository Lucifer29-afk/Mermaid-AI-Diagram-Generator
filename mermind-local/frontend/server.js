
// frontend/server.js
const express = require('express');
const path = require('path');
const app = express();

// Serve static files from the 'public' directory relative to this file
app.use(express.static(path.join(__dirname, 'public')));

const PORT = 3000;
app.listen(PORT, () => console.log(`Frontend server running at http://localhost:${PORT}`));
