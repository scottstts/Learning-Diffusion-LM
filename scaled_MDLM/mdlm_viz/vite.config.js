import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn } from 'child_process'

// Custom middleware to bridge to Python
const pythonInference = () => {
  return {
    name: 'python-inference',
    configureServer(server) {
      server.middlewares.use('/api/generate', (req, res, next) => {
        if (req.method === 'POST') {
          let body = ''
          req.on('data', chunk => {
            body += chunk.toString()
          })
          req.on('end', () => {
            // Spawn python process
            // We assume 'python' is in PATH. If using venv, might need specific path.
            // Using 'python3' for safe measure on Mac/Linux if 'python' is 2.x, but usually 'python' is fine or alias.
            const pythonProcess = spawn('python', ['backend/inference.py'], {
              stdio: ['pipe', 'pipe', 'pipe']
            })

            let output = ''
            let errorOutput = ''

            pythonProcess.stdout.on('data', (data) => {
              output += data.toString()
            })

            pythonProcess.stderr.on('data', (data) => {
              errorOutput += data.toString()
              console.error(`[Python Code]: ${data}`)
            })

            pythonProcess.on('close', (code) => {
              if (code !== 0) {
                res.statusCode = 500
                res.end(JSON.stringify({ error: 'Python script failed', details: errorOutput }))
              } else {
                res.setHeader('Content-Type', 'application/json')
                res.end(output)
              }
            })

            // Send input args to python
            pythonProcess.stdin.write(body)
            pythonProcess.stdin.end()
          })
        } else {
          next()
        }
      })
    }
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), pythonInference()],
})
