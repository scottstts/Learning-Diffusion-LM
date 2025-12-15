import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn } from 'child_process'

// Helper function to call Python inference script
const callPython = (body, res) => {
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

  pythonProcess.stdin.write(body)
  pythonProcess.stdin.end()
}

// Custom middleware to bridge to Python
const pythonInference = () => {
  return {
    name: 'python-inference',
    configureServer(server) {
      // Generate endpoint
      server.middlewares.use('/api/generate', (req, res, next) => {
        if (req.method === 'POST') {
          let body = ''
          req.on('data', chunk => { body += chunk.toString() })
          req.on('end', () => {
            // Inject mode into body
            const parsed = JSON.parse(body || '{}')
            parsed.mode = 'generate'
            callPython(JSON.stringify(parsed), res)
          })
        } else {
          next()
        }
      })

      // Tokenize endpoint
      server.middlewares.use('/api/tokenize', (req, res, next) => {
        if (req.method === 'POST') {
          let body = ''
          req.on('data', chunk => { body += chunk.toString() })
          req.on('end', () => {
            const parsed = JSON.parse(body || '{}')
            parsed.mode = 'tokenize'
            callPython(JSON.stringify(parsed), res)
          })
        } else {
          next()
        }
      })

      // Info endpoint (GET)
      server.middlewares.use('/api/info', (req, res, next) => {
        if (req.method === 'GET') {
          callPython(JSON.stringify({ mode: 'info' }), res)
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
